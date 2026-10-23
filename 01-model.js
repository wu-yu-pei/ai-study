import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import { tool, ToolMessage } from "langchain";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import path from "node:path";
import { z } from "zod";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));

dotenv.config()


const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

const getFileContent = tool(({ input }) => {
  const filePath = path.resolve(scriptDir, input);
  const result = fs.readFileSync(filePath, 'utf-8');
  return result;
}, {
  name: "getFileContent",
  description: "Get the content of a file",
  schema: z.object({
    input: z.string().describe("The path to the file"),
  }),
})

const tools = [
  getFileContent
]

const modelWithTools = model.bindTools(tools);


const messages = [
  { role: "user", content: "读取 ./package.json 的内容" },
];

let response = await modelWithTools.invoke(messages);
messages.push(response)
console.log(response)



while (response.tool_calls && response.tool_calls.length > 0) {
  const toolCallResults = await Promise.all(response.tool_calls.map(async toolCall => {
    const toolArgs = toolCall.args;
    const toolName = toolCall.name;
    const tool = tools.find(tool => tool.name === toolName);
    if (!tool) {
      throw new Error(`Tool ${toolName} not found`);
    }
    console.log(`Tool call: ${toolName} with input: ${JSON.stringify(toolArgs)}`);
    const result = await tool.invoke(toolArgs);
    console.log(`Tool result: ${result}`);
    return result;
  }));


  // 将工具结果添加到消息历史
  response.tool_calls.forEach((toolCall, index) => {
    messages.push(
      new ToolMessage({
        content: toolCallResults[index],
        tool_call_id: toolCall.id,
      })
    );
  });

  // 再次调用模型，传入工具结果
  response = await modelWithTools.invoke(messages);
}

console.log('\n[最终回复]');
console.log(response.content);