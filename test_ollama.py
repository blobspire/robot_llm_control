import ollama

def close_gripper() -> str:
    return "gripper_closed"

print("Beginning tool call test...")
print("Your output should resemble: \n\ncontent: \ntool_calls: [ToolCall(function=Function(name='close_gripper', arguments={}))]\n")

print("Initiating request to model...\nResults:\n")


resp = ollama.chat(
    model="gpt-oss:20b",
    messages=[
        {"role": "system", "content": "You must call the tool close_gripper. Do not write text."},
        {"role": "user", "content": "Close the gripper."},
    ],
    tools=[close_gripper],
    options={"temperature": 0.0},
)

print("content:", resp.message.content)
print("tool_calls:", resp.message.tool_calls)