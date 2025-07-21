# CHESS/src/workflow/agents/agent.py
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

from llm.models import call_engine, get_llm_chain
from llm.prompts import get_prompt
from runner.logger import Logger # Your custom logger
import logging # Standard Python logging for critical errors with tracebacks
import re

class Agent:
    """
    Abstract base class for agents.
    """
    
    def __init__(self, name: str, task: str, config: dict):
        self.name = name
        self.task = task
        self.config = config
        self.tools_config = config["tools"]
        self.tools = {}
        # self.chat_history is now built fresh for each LLM call within workout
    
    def workout(self, system_state: SystemState) -> SystemState:
        """
        Abstract method to process the system state.
        """
        
        # Load the base system prompt template
        system_prompt_template = get_prompt(template_name="agent_prompt")
        
        # This will store the turns of the conversation for the current workout
        current_conversation_history = [] 
        
        try:
            MAX_TOOL_CALLS_PER_WORKOUT = 10 
            for i in range(MAX_TOOL_CALLS_PER_WORKOUT):
                # 1. Format the chat history for the prompt
                chat_history_for_prompt_str = ""
                for chat_item in current_conversation_history:
                    role = chat_item["role"]
                    content = chat_item["content"]
                    chat_history_for_prompt_str += f"<{role}>\n{content}\n</{role}>\n"
                
                # 2. Construct the full prompt for this turn
                # This includes the static system instructions and the dynamic chat history
                full_prompt_for_llm = system_prompt_template.format(
                    agent_name=self.name,
                    task=self.task,
                    tools=self.get_tools_description(),
                    chat_history_string=chat_history_for_prompt_str # Pass formatted history
                )
                
                # 3. Call the LLM
                # Logger().log(f"Agent {self.name} sending prompt to LLM: '''{full_prompt_for_llm}'''", "debug") # Use your logger's debug
                logging.debug(f"Agent {self.name} sending prompt to LLM: '''{full_prompt_for_llm}'''")


                llm_response_content = self.call_agent_llm(full_prompt_for_llm)
                
                # Logger().log(f"Agent {self.name} LLM raw response: '''{llm_response_content}'''", "debug")
                logging.debug(f"Agent {self.name} LLM raw response: '''{llm_response_content}'''")

                # 4. Process response
                if self.is_done(llm_response_content):
                    # Logger().log(f"Agent {self.name} indicated DONE.", "info")
                    logging.info(f"Agent {self.name} indicated DONE.")
                    break
                
                tool_name = self.get_next_tool_name(llm_response_content)
                
                # Add LLM's "tool call" decision to conversation history
                current_conversation_history.append({"role": "assistant", "content": f"<tool_call>{tool_name}</tool_call>"})
                
                tool_to_call = self.tools[tool_name]
                try:
                    tool_call_result_message = self.call_tool(tool_to_call, system_state)
                    # Add tool's response to conversation history
                    current_conversation_history.append({"role": "tool_message", "content": tool_call_result_message})
                except Exception as e:
                    # Logger().log(f"Error executing tool {tool_name} for agent {self.name}: {e}", "error")
                    logging.error(f"Error executing tool {tool_name} for agent {self.name}: {e}", exc_info=True)
                    current_conversation_history.append({"role": "tool_message", "content": f"Error executing tool {tool_name}: {type(e).__name__} - {e}"})
            
            else: # Loop finished without break (max tool calls reached)
                # Logger().log(f"Agent {self.name} reached max tool calls ({MAX_TOOL_CALLS_PER_WORKOUT}) without 'DONE'.", "warning")
                logging.warning(f"Agent {self.name} reached max tool calls ({MAX_TOOL_CALLS_PER_WORKOUT}) without 'DONE'.")

        except Exception as e:
            # Logger().log(f"Error in agent {self.name} workout: {e}", "critical") # Or "error"
            logging.critical(f"Error in agent {self.name} workout: {e}", exc_info=True)
            
        return system_state

    def call_tool(self, tool: Tool, system_state: SystemState) -> str: # Return str message
        """
        Call a tool with the given name and system state.
        Returns a message string for the chat history.
        """
        try:
            # The tool itself modifies the system_state
            tool(system_state) 
            return f"Tool {tool.tool_name} executed successfully." # Simple success message
        except Exception as e:
            # Logger().log(f"Exception during call_tool for {tool.tool_name}: {e}", "error")
            logging.error(f"Exception during call_tool for {tool.tool_name}: {e}", exc_info=True)
            # Re-raise to be handled by the agent's main try-except or caller,
            # but also return an error message for history
            raise # Or return f"Error executing tool {tool.tool_name}: {type(e).__name__} - {e}" and don't re-raise
        
    def is_done(self, response: str) -> bool:
        """
        Check if the response indicates that the agent is done.
        Response should be *only* "DONE".
        """
        return response.strip() == "DONE"
    
    def get_next_tool_name(self, response: str) -> str:
        raw_response_for_log = response[:500] 
        try:
            cleaned_response = response.strip()
            
            # Attempt to find <tool_call>
            tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", cleaned_response, re.DOTALL | re.IGNORECASE)
            
            if tool_call_match:
                tool_name = tool_call_match.group(1).strip()
                if tool_name in self.tools:
                    logging.info(f"Agent {self.name} selected tool: {tool_name}")
                    return tool_name
                else:
                    logging.error(f"Agent {self.name}: LLM requested tool '{tool_name}' which is not in its toolkit {list(self.tools.keys())}. Raw response: '{raw_response_for_log}'")
                    raise ValueError(f"Tool '{tool_name}' not found in agent {self.name}'s toolkit.")
            # Fallback for models that return the tool name directly
            elif cleaned_response in self.tools:
                tool_name = cleaned_response
                logging.info(f"Agent {self.name} selected tool '{tool_name}' via raw response fallback.")
                return tool_name
            else:
                # This means the LLM output was not "DONE" and not a valid tool call.
                logging.error(f"Agent {self.name}: LLM response was not 'DONE' and did not contain a valid <tool_call> tag or raw tool name. Raw response: '{raw_response_for_log}'")
                raise ValueError(f"Agent {self.name}: Expected 'DONE' or a valid tool call, but got: '{raw_response_for_log}'")
        except Exception as e: 
            logging.critical(f"Agent {self.name}: Error parsing tool_call. Exception: {e}. Raw response: '{raw_response_for_log}'", exc_info=True)
            raise

    def call_agent_llm(self, full_prompt_str: str) -> str:
        """
        Directly calls the LLM with the fully constructed prompt string.
        """
        llm_chain_instance = get_llm_chain(engine_name=self.config["engine"], temperature=0)
        response_content = call_engine(message=full_prompt_str, engine=llm_chain_instance) # call_engine returns string content
        return response_content
        
    def get_tools_description(self) -> str:
        """
        Get the description of the tools.
        """
        tools_description = ""
        for i, (tool_key, tool_instance) in enumerate(self.tools.items()): # Iterate over items
            # Assuming tool_instance has a description or we just use the key
            # For now, just using the key as before.
            tools_description += f"{i+1}. {tool_key}\n" 
        return tools_description
    
    def __call__(self, system_state: SystemState) -> SystemState:
        return self.workout(system_state)