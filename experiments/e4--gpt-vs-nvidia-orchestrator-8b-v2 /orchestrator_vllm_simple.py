"""
Simple vLLM server for Orchestrator-8B
Run this on a GPU cloud instance (RunPod, Vast.ai, etc.)

Usage:
    python orchestrator_vllm_simple.py

Then access at: http://YOUR_POD_IP:8000/v1/chat/completions

Pre-downloading the model (if network issues):
    # Option 1: Pre-download to HuggingFace cache
    python -c "from transformers import AutoModel, AutoTokenizer; \
               AutoModel.from_pretrained('nvidia/Orchestrator-8B'); \
               AutoTokenizer.from_pretrained('nvidia/Orchestrator-8B')"
    
    # Option 2: Download to specific directory, then use env var
    export ORCHESTRATOR_MODEL_PATH=/path/to/downloaded/model
    python orchestrator_vllm_simple.py
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import uvicorn
import json
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize vLLM (loads model on startup)
# Support both local model path and HuggingFace model name
import os
from pathlib import Path

MODEL_NAME_ENV = os.getenv("ORCHESTRATOR_MODEL_PATH")
HF_MODEL_NAME = "nvidia/Orchestrator-8B"

# Check for local model path first, then try HuggingFace cache, then fallback to HF name
MODEL_NAME = None
if MODEL_NAME_ENV:
    # User specified a local path
    if Path(MODEL_NAME_ENV).exists():
        MODEL_NAME = MODEL_NAME_ENV
        print(f"✅ Using local model path: {MODEL_NAME}")
    else:
        print(f"⚠️  Specified path doesn't exist: {MODEL_NAME_ENV}")
        print(f"   Falling back to HuggingFace model: {HF_MODEL_NAME}")

if MODEL_NAME is None:
    # Try to find model in HuggingFace cache
    hf_cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    possible_cache_paths = [
        Path(hf_cache_dir) / "hub" / f"models--{HF_MODEL_NAME.replace('/', '--')}",
        Path(hf_cache_dir) / "transformers" / HF_MODEL_NAME.replace("/", "--"),
    ]
    
    for cache_path in possible_cache_paths:
        if cache_path.exists():
            # Find the actual model directory (might be in a snapshot subdirectory)
            snapshots = list(cache_path.glob("snapshots/*"))
            if snapshots:
                MODEL_NAME = str(snapshots[0])
                print(f"✅ Found model in HuggingFace cache: {MODEL_NAME}")
                break
    
    if MODEL_NAME is None:
        MODEL_NAME = HF_MODEL_NAME
        print(f"⚠️  Model not found in cache, will try to download from HuggingFace")
        print(f"   (This will fail if network is unavailable)")

print(f"🔄 Loading Orchestrator-8B with vLLM...")
print(f"   Model: {MODEL_NAME}")

try:
    # Try to load model - if it fails due to network, provide helpful error
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,  # Use 1 GPU (increase for multi-GPU)
        gpu_memory_utilization=0.9,
        max_model_len=8192,  # Increased to handle longer prompts with many tools
        download_dir=None,  # Use default cache directory
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✅ Model loaded successfully!")
except Exception as e:
    error_msg = str(e).lower()
    if "name resolution" in error_msg or "failed to resolve" in error_msg or "temporary failure" in error_msg:
        print(f"\n❌ ERROR: Cannot reach HuggingFace (DNS/Network failure)")
        print(f"\n💡 SOLUTION: You need to use a pre-downloaded model.")
        print(f"\n   Option 1: Download model on a machine with internet, then copy to server:")
        print(f"     1. On a machine with internet, run:")
        print(f"        python -c \"from transformers import AutoModel, AutoTokenizer; \\")
        print(f"                     AutoModel.from_pretrained('{HF_MODEL_NAME}'); \\")
        print(f"                     AutoTokenizer.from_pretrained('{HF_MODEL_NAME}')\"")
        print(f"     2. Copy the cached model to your server:")
        print(f"        scp -r ~/.cache/huggingface/hub/models--nvidia--Orchestrator-8B user@server:/path/")
        print(f"     3. Set environment variable:")
        print(f"        export ORCHESTRATOR_MODEL_PATH=/path/to/models--nvidia--Orchestrator-8B/snapshots/[hash]")
        print(f"\n   Option 2: If model is already on server, set path:")
        print(f"        export ORCHESTRATOR_MODEL_PATH=/full/path/to/model/directory")
        print(f"\n   Option 3: Fix DNS/Network on server:")
        print(f"        - Check DNS: nslookup huggingface.co")
        print(f"        - Try different DNS: echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf")
        print(f"        - Check firewall/proxy settings")
    else:
        print(f"\n❌ ERROR: Failed to load model: {e}")
        print(f"\n💡 Check the error above and ensure:")
        print(f"   - Model path is correct")
        print(f"   - GPU is available")
        print(f"   - Sufficient GPU memory")
    raise

class Message(BaseModel):
    role: str
    content: Optional[str] = None  # Can be None for tool call responses
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Tool calls if present
    tool_call_id: Optional[str] = None  # Tool call ID for tool responses

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.2
    max_tokens: int = 4096
    model: Optional[str] = None  # OpenAI compatibility
    tools: Optional[List[Dict[str, Any]]] = None  # Tools for function calling
    tool_choice: Optional[str] = None  # Tool choice mode

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    try:
        # Log incoming request (reduced verbosity for performance)
        logger.debug(f"Received request with {len(request.messages)} messages")
        if request.tools:
            logger.debug(f"Tools provided: {len(request.tools)} tools")
        
        # Build prompt with tool information if provided
        # Handle messages with None content (tool call responses)
        messages_for_template = []
        for m in request.messages:
            # Handle different message types:
            # 1. Assistant messages with tool_calls: content can be None (valid)
            # 2. Tool response messages (role="tool"): should have content (tool result)
            # 3. Regular messages: should have content
            
            # Determine content value
            if m.content is not None:
                content = m.content
            elif m.role == "assistant" and m.tool_calls:
                # Assistant tool call messages can have None content - use empty string
                content = ""
            elif m.role == "tool":
                # Tool response messages should have content, but if None, use empty string
                content = ""
                logger.warning(f"Tool response message has None content, using empty string")
            else:
                # Other messages with None content are invalid
                logger.warning(f"Skipping invalid message: role={m.role}, content=None, tool_calls={bool(m.tool_calls)}")
                continue
            
            # Build message dict
            msg_dict = {"role": m.role, "content": content}
            
            # Add tool_call_id if present (for tool responses)
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            
            messages_for_template.append(msg_dict)
        
        # If tools are provided, add them to the system message or user message
        if request.tools:
            # Create concise tool descriptions to avoid exceeding token limits
            tools_description = "\n\nYou have access to these tools. Respond with JSON: {\"tool_name\": \"name\", \"arguments\": {...}}\n\nTools:\n"
            
            # Limit tool description length to prevent prompt overflow
            MAX_TOOL_DESC_LENGTH = 200  # Max chars per tool description
            MAX_PARAM_DESC_LENGTH = 50  # Max chars per parameter description
            
            for tool in request.tools:
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                # Truncate description if too long
                if len(desc) > MAX_TOOL_DESC_LENGTH:
                    desc = desc[:MAX_TOOL_DESC_LENGTH] + "..."
                
                params = func.get("parameters", {})
                
                # Compact format: ToolName: description (param1:type, param2:type)
                param_list = []
                if params:
                    props = params.get("properties", {})
                    required = params.get("required", [])
                    for param_name, param_info in props.items():
                        param_type = param_info.get("type", "string")
                        is_req = param_name in required
                        param_marker = "*" if is_req else ""
                        param_list.append(f"{param_name}{param_marker}:{param_type}")
                
                param_str = f" ({', '.join(param_list)})" if param_list else ""
                tools_description += f"- {name}: {desc}{param_str}\n"
            
            tools_description += "\nRespond with JSON tool call: {\"tool_name\": \"name\", \"arguments\": {...}}\n"
            
            # Estimate token count and truncate if needed
            estimated_tokens = len(tokenizer.encode(tools_description))
            MAX_TOOL_DESCRIPTION_TOKENS = 1200  # Reduced from 2000 - more aggressive limit to match reduced system message size
            
            if estimated_tokens > MAX_TOOL_DESCRIPTION_TOKENS:
                logger.warning(f"Tool description too long ({estimated_tokens} tokens), truncating...")
                # Truncate by removing tools from the end
                lines = tools_description.split('\n')
                header_lines = 3  # Keep header lines
                tool_lines = [l for l in lines[header_lines:-1] if l.startswith('-')]  # Tool lines
                footer = lines[-1]  # Keep footer
                
                # Keep only first N tools that fit
                truncated_tools = []
                current_tokens = len(tokenizer.encode('\n'.join(lines[:header_lines] + [footer])))
                for tool_line in tool_lines:
                    line_tokens = len(tokenizer.encode(tool_line))
                    if current_tokens + line_tokens > MAX_TOOL_DESCRIPTION_TOKENS:
                        break
                    truncated_tools.append(tool_line)
                    current_tokens += line_tokens
                
                tools_description = '\n'.join(lines[:header_lines] + truncated_tools + [footer])
                logger.info(f"Truncated to {len(truncated_tools)} tools, ~{current_tokens} tokens")
            
            # Add tools info as a system message
            if messages_for_template and messages_for_template[0].get("role") != "system":
                messages_for_template.insert(0, {"role": "system", "content": tools_description})
            elif messages_for_template:
                # Prepend to existing system message
                if messages_for_template[0]["role"] == "system":
                    messages_for_template[0]["content"] = tools_description + "\n\n" + messages_for_template[0]["content"]
                else:
                    messages_for_template.insert(0, {"role": "system", "content": tools_description})
            else:
                messages_for_template.insert(0, {"role": "system", "content": tools_description})
        
        # Format messages using chat template
        if tokenizer.chat_template:
            formatted_prompt = tokenizer.apply_chat_template(
                messages_for_template,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages_for_template])
        
        # Check total prompt length and truncate if needed
        prompt_tokens = len(tokenizer.encode(formatted_prompt))
        MAX_PROMPT_TOKENS = 7000  # Increased from 6000 - use more of the 8192 model limit (leaves ~1000 for response)
        
        if prompt_tokens > MAX_PROMPT_TOKENS:
            logger.warning(f"Prompt too long ({prompt_tokens} tokens), truncating conversation history...")
            
            # Improved truncation strategy:
            # 1. Always keep system message (but truncate if needed)
            # 2. Always keep the most recent user message
            # 3. Always keep the most recent assistant message (if exists)
            # 4. Truncate tool outputs more aggressively (they can be huge)
            # 5. Remove oldest messages first
            
            system_msg = None
            other_messages = []
            
            for msg in messages_for_template:
                if msg.get("role") == "system":
                    system_msg = msg
                else:
                    other_messages.append(msg)
            
            # Calculate system message tokens and truncate if needed
            if system_msg:
                system_content = system_msg.get("content", "")
                system_tokens = len(tokenizer.encode(system_content))
                MAX_SYSTEM_TOKENS = 1500  # Reduced from 2500 - more aggressive limit to leave room for conversation
                if system_tokens > MAX_SYSTEM_TOKENS:
                    tokens = tokenizer.encode(system_content)
                    # Keep first part (tool descriptions) and last part (instructions)
                    # Split roughly in half, keeping both ends
                    mid_point = len(tokens) // 2
                    keep_start = tokens[:MAX_SYSTEM_TOKENS // 2]
                    keep_end = tokens[-MAX_SYSTEM_TOKENS // 2:]
                    truncated_tokens = keep_start + keep_end
                    system_content = tokenizer.decode(truncated_tokens)
                    system_msg = system_msg.copy()
                    system_msg['content'] = system_content
                    system_tokens = len(truncated_tokens)
                    logger.info(f"Truncated system message from {len(tokens)} to {len(truncated_tokens)} tokens")
            else:
                system_tokens = 0
            
            logger.info(f"System message size: {system_tokens} tokens (max: {MAX_SYSTEM_TOKENS})")
            
            # Separate messages by type for smarter truncation
            user_messages = [msg for msg in other_messages if msg.get("role") == "user"]
            assistant_messages = [msg for msg in other_messages if msg.get("role") == "assistant"]
            tool_messages = [msg for msg in other_messages if msg.get("role") == "tool"]
            
            # Token limits per message type
            MAX_USER_MESSAGE_TOKENS = 1000  # User messages are important, allow more
            MAX_ASSISTANT_MESSAGE_TOKENS = 1000  # Assistant messages are important
            MAX_TOOL_MESSAGE_TOKENS = 1500  # Tool outputs can be huge, but we need some context
            
            def summarize_message_turn(messages: List[Dict], max_summary_tokens: int = 200) -> Dict:
                """
                Summarize a conversation turn (user -> assistant -> tool responses) into a compact summary.
                This is the KEY optimization - preserves context while dramatically reducing tokens.
                
                Returns a summary message dict with role='assistant' and summarized content.
                """
                if not messages:
                    return None
                
                user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                assistant_msg = next((m for m in messages if m.get('role') == 'assistant'), None)
                tool_msgs = [m for m in messages if m.get('role') == 'tool']
                
                summary_parts = []
                
                # Extract user intent (keep original if short, summarize if long)
                if user_msg:
                    user_content = user_msg.get('content', '')
                    user_tokens = len(tokenizer.encode(user_content))
                    if user_tokens <= 100:
                        summary_parts.append(f"User: {user_content}")
                    else:
                        # Summarize long user messages
                        tokens = tokenizer.encode(user_content)
                        summary_tokens = tokens[:50]  # Keep first 50 tokens
                        summary_parts.append(f"User: {tokenizer.decode(summary_tokens)}...")
                
                # Extract assistant action (tool calls or conclusion)
                if assistant_msg:
                    assistant_content = assistant_msg.get('content', '')
                    tool_calls = assistant_msg.get('tool_calls', [])
                    
                    if tool_calls:
                        # Extract tool call names
                        tool_names = []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tool_name = tc.get('function', {}).get('name', '') or tc.get('name', '')
                            else:
                                tool_name = getattr(tc, 'name', '')
                            if tool_name:
                                tool_names.append(tool_name)
                        
                        if tool_names:
                            summary_parts.append(f"Assistant called: {', '.join(tool_names[:3])}")
                        else:
                            summary_parts.append("Assistant made tool calls")
                    elif assistant_content:
                        # Extract conclusion (last sentence or first 100 tokens)
                        tokens = tokenizer.encode(assistant_content)
                        if len(tokens) > 100:
                            conclusion_tokens = tokens[-100:]  # Last 100 tokens (conclusion)
                            summary_parts.append(f"Assistant: ...{tokenizer.decode(conclusion_tokens)}")
                        else:
                            summary_parts.append(f"Assistant: {assistant_content}")
                
                # Summarize tool outputs (extract success/failure and key data)
                if tool_msgs:
                    tool_summaries = []
                    for tool_msg in tool_msgs[:3]:  # Limit to 3 most recent tool outputs
                        tool_content = str(tool_msg.get('content', ''))
                        if not tool_content:
                            continue
                        
                        # Try to extract key information
                        tokens = tokenizer.encode(tool_content)
                        if len(tokens) > 150:
                            # Extract beginning (status) and end (result)
                            start_tokens = tokens[:50]
                            end_tokens = tokens[-50:]
                            summary = f"{tokenizer.decode(start_tokens)}...{tokenizer.decode(end_tokens)}"
                        else:
                            summary = tool_content
                        
                        # Check for success/failure indicators
                        content_lower = tool_content.lower()
                        if 'success' in content_lower or '200' in tool_content or 'ok' in content_lower:
                            tool_summaries.append(f"✓ Tool succeeded: {summary[:100]}")
                        elif 'error' in content_lower or 'fail' in content_lower or '404' in tool_content:
                            tool_summaries.append(f"✗ Tool failed: {summary[:100]}")
                        else:
                            tool_summaries.append(f"Tool output: {summary[:100]}")
                    
                    if tool_summaries:
                        summary_parts.append(" | ".join(tool_summaries))
                
                # Combine into summary
                summary_text = " | ".join(summary_parts)
                
                # Truncate if still too long
                summary_tokens = len(tokenizer.encode(summary_text))
                if summary_tokens > max_summary_tokens:
                    tokens = tokenizer.encode(summary_text)
                    truncated = tokens[:max_summary_tokens]
                    summary_text = tokenizer.decode(truncated) + "..."
                
                return {
                    "role": "assistant",
                    "content": f"[Summarized turn] {summary_text}"
                }
            
            def truncate_message_content(content: str, max_tokens: int, role: str) -> str:
                """Truncate message content intelligently."""
                if not content:
                    return content
                
                tokens = tokenizer.encode(content)
                if len(tokens) <= max_tokens:
                    return content
                
                # For tool messages, keep beginning (summary) and end (most recent data)
                # For other messages, keep beginning and end
                if role == "tool" and len(tokens) > max_tokens * 2:
                    # Tool output is very long - keep first 30% and last 70%
                    keep_start_tokens = max_tokens // 3
                    keep_end_tokens = max_tokens - keep_start_tokens
                    truncated = tokens[:keep_start_tokens] + tokens[-keep_end_tokens:]
                    truncated_content = tokenizer.decode(truncated)
                    logger.debug(f"Truncated {role} message: {len(tokens)} -> {len(truncated)} tokens (kept start+end)")
                    return truncated_content
                else:
                    # Keep last N tokens (most recent information)
                    truncated = tokens[-max_tokens:]
                    truncated_content = tokenizer.decode(truncated)
                    logger.debug(f"Truncated {role} message: {len(tokens)} -> {len(truncated)} tokens (kept end)")
                    return truncated_content
            
            # CRITICAL: Message summarization strategy
            # Group messages into "turns" (user -> assistant -> tool responses)
            # Keep recent turns in full detail, summarize older turns
            
            # Identify which messages to keep in full detail (recent turns)
            RECENT_TURNS_TO_KEEP = 2  # Keep last 2 turns in full detail
            
            messages_to_keep_full = set()
            
            # Always keep the most recent user message
            if user_messages:
                messages_to_keep_full.add(id(user_messages[-1]))
            
            # Keep recent assistant messages (up to 2 most recent turns)
            for msg in assistant_messages[-RECENT_TURNS_TO_KEEP:]:
                messages_to_keep_full.add(id(msg))
            
            # Keep tool messages associated with recent turns (up to 5 most recent)
            for msg in tool_messages[-5:]:
                messages_to_keep_full.add(id(msg))
            
            # Group messages into turns for summarization
            # A turn starts with a user message and includes all assistant/tool messages until next user message
            turns = []
            current_turn = []
            
            for msg in other_messages:
                role = msg.get('role', '')
                if role == 'user' and current_turn:
                    # Start new turn
                    turns.append(current_turn)
                    current_turn = [msg]
                else:
                    current_turn.append(msg)
            
            if current_turn:
                turns.append(current_turn)
            
            # Process turns: summarize old ones, keep recent ones in full
            kept_messages = []
            current_tokens = system_tokens
            recent_turns = turns[-RECENT_TURNS_TO_KEEP:] if len(turns) > RECENT_TURNS_TO_KEEP else turns
            old_turns = turns[:-RECENT_TURNS_TO_KEEP] if len(turns) > RECENT_TURNS_TO_KEEP else []
            
            # First, add summaries of old turns (if any)
            for turn in old_turns:
                summary = summarize_message_turn(turn, max_summary_tokens=150)
                if summary:
                    summary_tokens = len(tokenizer.encode(f"assistant: {summary.get('content', '')}"))
                    if current_tokens + summary_tokens <= MAX_PROMPT_TOKENS - 2000:  # Reserve space for recent turns
                        kept_messages.append(summary)
                        current_tokens += summary_tokens
                        logger.debug(f"Added summary of old turn ({summary_tokens} tokens)")
            
            # Then, add recent turns in full detail (with truncation if needed)
            for turn in recent_turns:
                for msg in turn:
                    msg_copy = msg.copy()
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    
                    # Determine max tokens based on role
                    if role == "user":
                        max_tokens = MAX_USER_MESSAGE_TOKENS
                    elif role == "assistant":
                        max_tokens = MAX_ASSISTANT_MESSAGE_TOKENS
                    elif role == "tool":
                        max_tokens = MAX_TOOL_MESSAGE_TOKENS
                    else:
                        max_tokens = 500  # Default
                    
                    # Truncate if needed
                    msg_copy['content'] = truncate_message_content(content, max_tokens, role)
                    
                    # Calculate tokens for this message
                    msg_tokens = len(tokenizer.encode(f"{role}: {msg_copy.get('content', '')}"))
                    
                    # Check if we can fit this message
                    if current_tokens + msg_tokens > MAX_PROMPT_TOKENS:
                        logger.warning(f"Cannot fit all recent turn messages, stopping at {len(kept_messages)} messages")
                        break
                    
                    kept_messages.append(msg_copy)
                    current_tokens += msg_tokens
                
                # Break outer loop if we've exceeded token limit
                if current_tokens > MAX_PROMPT_TOKENS - 500:  # Leave some buffer
                    break
            
            # Rebuild messages in chronological order (system, then conversation)
            messages_for_template = []
            if system_msg:
                messages_for_template.append(system_msg)
            messages_for_template.extend(kept_messages)
            
            # Reformat prompt
            if tokenizer.chat_template:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages_for_template,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages_for_template])
            
            final_tokens = len(tokenizer.encode(formatted_prompt))
            conversation_tokens = final_tokens - system_tokens
            
            # Enhanced logging: breakdown by message type
            user_count = len([m for m in kept_messages if m.get('role') == 'user'])
            assistant_count = len([m for m in kept_messages if m.get('role') == 'assistant'])
            tool_count = len([m for m in kept_messages if m.get('role') == 'tool'])
            summary_count = len([m for m in kept_messages if '[Summarized turn]' in str(m.get('content', ''))])
            
            logger.info(f"Truncated prompt: {prompt_tokens} -> {final_tokens} tokens")
            logger.info(f"  Token breakdown: system={system_tokens}, conversation={conversation_tokens}, total={final_tokens}")
            logger.info(f"  Message breakdown: {user_count} user, {assistant_count} assistant, {tool_count} tool, {summary_count} summarized turns")
            logger.info(f"  Kept {len(kept_messages)} conversation messages ({len(old_turns)} old turns summarized, {len(recent_turns)} recent turns in full)")
        
        # Final safety check - hard truncate if still too long
        final_prompt_tokens = len(tokenizer.encode(formatted_prompt))
        ABSOLUTE_MAX_TOKENS = 7500  # Hard limit - must be under max_model_len
        
        if final_prompt_tokens > ABSOLUTE_MAX_TOKENS:
            logger.error(f"Prompt still too long after truncation ({final_prompt_tokens} tokens), hard truncating...")
            # Hard truncate the prompt itself (keep first N tokens)
            tokens = tokenizer.encode(formatted_prompt)
            truncated_tokens = tokens[:ABSOLUTE_MAX_TOKENS]
            formatted_prompt = tokenizer.decode(truncated_tokens)
            final_prompt_tokens = len(truncated_tokens)
            logger.warning(f"Hard truncated prompt to {final_prompt_tokens} tokens")
        
        # Generate with vLLM
        # Increased max_tokens limit to support reasoning traces (System 2 thinking)
        # Reasoning models need more tokens to generate thorough thinking before tool calls
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=min(request.max_tokens, 8000),  # Increased from 1000 to 8000 for reasoning traces
            skip_special_tokens=True,  # Skip special tokens for cleaner output
        )
        
        outputs = llm.generate([formatted_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        # Strip reasoning tokens before parsing (same logic as ReasoningTokenStripper)
        # Remove reasoning tags that might wrap the actual response
        cleaned_response = response_text
        if request.tools:
            # Remove reasoning tags and their content (order matters - do paired tags first)
            # Handle various reasoning tag formats (Orchestrator uses these)
            cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            cleaned_response = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            # Remove any remaining standalone tags (in case of unclosed tags)
            cleaned_response = re.sub(r'</?redacted_reasoning>', '', cleaned_response, flags=re.IGNORECASE)
            cleaned_response = re.sub(r'</?think>', '', cleaned_response, flags=re.IGNORECASE)
            cleaned_response = re.sub(r'</?reasoning>', '', cleaned_response, flags=re.IGNORECASE)
            # Clean up extra whitespace
            cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)
            cleaned_response = cleaned_response.strip()
        
        # Log response for debugging (only if verbose or if cleaning changed response)
        if request.tools and cleaned_response != response_text:
            logger.debug(f"Cleaned reasoning tags from response (removed {len(response_text) - len(cleaned_response)} chars)")
        
        # Try to parse tool calls from response text
        tool_calls = []
        json_patterns = []  # Initialize at function scope to avoid scoping issues
        if request.tools:
            import json
            
            # Create tool name map once for efficient lookups (O(1) instead of O(n))
            tool_name_map = {tool.get("function", {}).get("name"): tool for tool in request.tools}
            
            # Use cleaned response for parsing
            parse_text = cleaned_response
            
            # First, try to parse the entire response as JSON (most common case)
            parsed_tool_call = None
            try:
                stripped = parse_text.strip()
                if stripped.startswith('{'):
                    tool_call_data = json.loads(stripped)
                    if "tool_name" in tool_call_data:
                        parsed_tool_call = tool_call_data
                        logger.debug(f"✅ Parsed full response as JSON tool call: {tool_call_data.get('tool_name')}")
            except json.JSONDecodeError:
                # Not valid JSON, try to find JSON objects in the text
                logger.debug("Full response is not valid JSON, will try pattern matching")
                pass
            
            # If we found a tool call in the full response, use it
            if parsed_tool_call:
                tool_name = parsed_tool_call.get("tool_name")
                if tool_name:
                    # Optimized: use pre-built map instead of linear search
                    matching_tool = tool_name_map.get(tool_name)
                    
                    if matching_tool:
                        arguments = parsed_tool_call.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except:
                                arguments = {}
                        
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments
                            }
                        })
                        logger.debug(f"✅ Parsed tool call from full response: {tool_name}")
            
            # If no tool call found yet, try to find JSON objects in the text (for multi-tool scenarios)
            if not tool_calls:
                # Use a more sophisticated approach to find JSON objects with nested braces
                # Find all potential JSON object boundaries
                start_idx = 0
                while True:
                    # Find the start of a JSON object
                    start = parse_text.find('{', start_idx)
                    if start == -1:
                        break
                    
                    # Try to find the matching closing brace
                    brace_count = 0
                    end = start
                    for i in range(start, len(parse_text)):
                        if parse_text[i] == '{':
                            brace_count += 1
                        elif parse_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    if end > start:
                        potential_json = parse_text[start:end]
                        json_patterns.append(potential_json)
                        start_idx = end
                    else:
                        start_idx = start + 1
            
            # Try to parse each potential JSON pattern (only if we have patterns and no tool calls yet)
            if json_patterns and not tool_calls:
                for pattern in json_patterns:
                    try:
                        # Try to parse as JSON
                        tool_call_data = json.loads(pattern)
                        
                        # Check for tool_name field
                        tool_name = None
                        if "tool_name" in tool_call_data:
                            tool_name = tool_call_data["tool_name"]
                        elif "function" in tool_call_data and isinstance(tool_call_data["function"], dict):
                            tool_name = tool_call_data["function"].get("name")
                        elif "name" in tool_call_data:
                            tool_name = tool_call_data["name"]
                        
                        if tool_name:
                            # Optimized: use pre-built map instead of linear search
                            matching_tool = tool_name_map.get(tool_name)
                            
                            if matching_tool:
                                # Extract arguments
                                arguments = {}
                                if "arguments" in tool_call_data:
                                    if isinstance(tool_call_data["arguments"], dict):
                                        arguments = tool_call_data["arguments"]
                                    elif isinstance(tool_call_data["arguments"], str):
                                        try:
                                            arguments = json.loads(tool_call_data["arguments"])
                                        except:
                                            arguments = {}
                                
                                tool_calls.append({
                                    "id": f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else (arguments if arguments else "{}")
                                    }
                                })
                                logger.debug(f"✅ Parsed tool call from pattern: {tool_name}")
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        try:
                            # Remove trailing commas
                            fixed_pattern = re.sub(r',\s*}', '}', pattern)
                            fixed_pattern = re.sub(r',\s*]', ']', fixed_pattern)
                            tool_call_data = json.loads(fixed_pattern)
                            # Same parsing logic as above...
                        except:
                            pass
                    except Exception as e:
                        logger.debug(f"Error parsing tool call pattern: {e}")
                        pass
            
        
        # Estimate tokens (cache tokenizer calls for efficiency)
        prompt_tokens = len(tokenizer.encode(formatted_prompt))
        completion_tokens = len(tokenizer.encode(response_text))
        
        # Build response message
        # If we have tool_calls, set content to None (OpenAI format)
        # Otherwise, use the response text as content
        message_content = None if tool_calls else response_text
        
        # Return OpenAI-compatible format
        response = {
            "id": "chatcmpl-orchestrator",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message_content if not tool_calls else None,
                        "tool_calls": tool_calls if tool_calls else None
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
        logger.debug(f"Response: {len(response_text)} chars, {completion_tokens} tokens, {len(tool_calls)} tool calls")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        # Log the raw response for debugging
        error_response = {
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "raw_response": None
            }
        }
        return JSONResponse(content=error_response, status_code=500)

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME, "engine": "vLLM"}

if __name__ == "__main__":
    print("🚀 Starting vLLM server on port 8000...")
    print("📡 API available at: http://0.0.0.0:8000/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8000)

