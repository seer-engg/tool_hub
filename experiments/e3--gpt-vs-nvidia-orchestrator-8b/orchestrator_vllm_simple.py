"""
Simple vLLM server for Orchestrator-8B
Run this on a GPU cloud instance (RunPod, Vast.ai, etc.)

Usage:
    python orchestrator_vllm_simple.py

Then access at: http://YOUR_POD_IP:8000/v1/chat/completions
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
print("🔄 Loading Orchestrator-8B with vLLM...")
MODEL_NAME = "nvidia/Orchestrator-8B"  # Official NVIDIA model

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,  # Use 1 GPU (increase for multi-GPU)
    gpu_memory_utilization=0.9,
    max_model_len=8192,  # Increased to handle longer prompts with many tools
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✅ Model loaded successfully!")

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
        # Log incoming request
        logger.info(f"Received request with {len(request.messages)} messages")
        if request.tools:
            logger.info(f"Tools provided: {len(request.tools)} tools")
            for tool in request.tools:
                func_name = tool.get("function", {}).get("name", "unknown")
                logger.info(f"  - {func_name}")
        
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
            MAX_TOOL_DESCRIPTION_TOKENS = 2000  # More aggressive limit - reserve space for user messages and response
            
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
        MAX_PROMPT_TOKENS = 6000  # More aggressive limit - leave room for response generation (8192 - 2000 for response)
        
        if prompt_tokens > MAX_PROMPT_TOKENS:
            logger.warning(f"Prompt too long ({prompt_tokens} tokens), truncating conversation history...")
            
            # Truncate by removing oldest messages (keep system message and recent messages)
            # Keep system message and last N messages that fit
            system_msg = None
            other_messages = []
            
            for msg in messages_for_template:
                if msg.get("role") == "system":
                    system_msg = msg
                else:
                    other_messages.append(msg)
            
            # Calculate system message tokens
            if system_msg:
                system_content = system_msg.get("content", "")
                system_tokens = len(tokenizer.encode(system_content))
            else:
                system_tokens = 0
            
            # Keep only recent messages that fit
            kept_messages = []
            current_tokens = system_tokens
            
            # Process messages in reverse (keep most recent)
            # Also truncate individual message content if too long
            MAX_MESSAGE_TOKENS = 500  # Max tokens per message
            
            for msg in reversed(other_messages):
                msg_content = msg.get('content', '')
                
                # Truncate individual message if too long
                msg_tokens_raw = len(tokenizer.encode(msg_content))
                if msg_tokens_raw > MAX_MESSAGE_TOKENS:
                    # Truncate message content (keep last N tokens)
                    tokens = tokenizer.encode(msg_content)
                    truncated_tokens = tokens[-MAX_MESSAGE_TOKENS:]  # Keep last N tokens
                    msg_content = tokenizer.decode(truncated_tokens)
                    msg = msg.copy()
                    msg['content'] = msg_content
                    logger.debug(f"Truncated message from {msg_tokens_raw} to {MAX_MESSAGE_TOKENS} tokens")
                
                msg_tokens = len(tokenizer.encode(f"{msg.get('role')}: {msg_content}"))
                
                if current_tokens + msg_tokens > MAX_PROMPT_TOKENS:
                    logger.warning(f"Stopping truncation: current={current_tokens}, adding={msg_tokens}, max={MAX_PROMPT_TOKENS}")
                    break
                
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens
            
            # Rebuild messages
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
            logger.info(f"Truncated prompt: {prompt_tokens} -> {final_tokens} tokens, kept {len(kept_messages)} messages")
        
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
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=min(request.max_tokens, 1000),  # Limit response tokens to prevent issues
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
            cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            cleaned_response = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
            # Remove any remaining standalone tags (in case of unclosed tags)
            cleaned_response = re.sub(r'</?redacted_reasoning>', '', cleaned_response, flags=re.IGNORECASE)
            cleaned_response = re.sub(r'</?think>', '', cleaned_response, flags=re.IGNORECASE)
            cleaned_response = re.sub(r'</?reasoning>', '', cleaned_response, flags=re.IGNORECASE)
            # Clean up extra whitespace
            cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)
            cleaned_response = cleaned_response.strip()
        
        # Log response for debugging
        if request.tools:
            logger.info(f"Response text (first 500 chars): {response_text[:500]}...")
            if cleaned_response != response_text:
                logger.info(f"Cleaned response (first 500 chars): {cleaned_response[:500]}...")
        
        # Try to parse tool calls from response text
        tool_calls = []
        json_patterns = []  # Initialize at function scope to avoid scoping issues
        if request.tools:
            import json
            
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
                        logger.info(f"✅ Parsed full response as JSON tool call: {tool_call_data}")
            except json.JSONDecodeError:
                # Not valid JSON, try to find JSON objects in the text
                logger.debug("Full response is not valid JSON, will try pattern matching")
                pass
            
            # If we found a tool call in the full response, use it
            if parsed_tool_call:
                tool_name = parsed_tool_call.get("tool_name")
                if tool_name:
                    matching_tool = None
                    for tool in request.tools:
                        if tool.get("function", {}).get("name") == tool_name:
                            matching_tool = tool
                            break
                    
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
                        logger.info(f"✅ Parsed tool call from full response: {tool_name} with args: {arguments}")
            
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
                            # Find matching tool
                            matching_tool = None
                            for tool in request.tools:
                                if tool.get("function", {}).get("name") == tool_name:
                                    matching_tool = tool
                                    break
                            
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
                                logger.info(f"✅ Parsed tool call from pattern: {tool_name} with args: {arguments}")
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
            
        
        # Estimate tokens
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
        
        logger.info(f"Response generated: {len(response_text)} chars, {completion_tokens} tokens, {len(tool_calls)} tool calls")
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

