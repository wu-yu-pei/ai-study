import dotenv from "dotenv";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage, tool, ToolMessage } from "langchain";
import { spawn } from 'child_process';
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import path from "node:path";
import { z } from "zod";


// 加载环境变量
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(scriptDir, ".env"), quiet: true });

// 初始化模型
const model = new ChatOpenAI({
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
})

// tools
// 读取文件
const readFile = tool(
  ({ filePath }) => {
    filePath = path.resolve(scriptDir, filePath);
    const result = fs.readFileSync(filePath, 'utf-8');
    return result;
  },
  {
    name: "readFile",
    description: "Read the content of a file",
    schema: z.object({
      filePath: z.string().describe("The path to the file"),
    }),
  })

// 写入文件
const writeFile = tool(
  ({ filePath, fileContent }) => {
    filePath = path.resolve(scriptDir, filePath);
    fs.writeFileSync(filePath, fileContent, 'utf-8');
    return 'success';
  },
  {
    name: "writeFile",
    description: "Write the content of a file",
    schema: z.object({
      filePath: z.string().describe("The path to the file"),
      fileContent: z.string().describe("The content to write to the file"),
    }),
  })

// 列出目录
const listDir = tool(
  ({ dirPath }) => {
    dirPath = path.resolve(scriptDir, dirPath);
    const result = fs.readdirSync(dirPath);
    return result;
  },
  {
    name: "listDir",
    description: "List the content of a directory",
    schema: z.object({
      dirPath: z.string().describe("The path to the directory"),
    }),
  }
)
// 执行命令
const execCommand = tool(
  ({ command }) =>
    new Promise((resolve, reject) => {
      const child = spawn(command, {
        shell: true,
        cwd: scriptDir,
        windowsHide: true,
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', chunk => {
        stdout += chunk.toString();
      });

      child.stderr.on('data', chunk => {
        stderr += chunk.toString();
      });

      child.on('error', reject);

      child.on('close', code => {
        resolve(JSON.stringify({
          exitCode: code,
          stdout,
          stderr,
        }));
      });
    }),
  {
    name: 'execCommand',
    description: 'Execute a command and return its exit code, stdout and stderr',
    schema: z.object({
      command: z.string().describe('The command to execute'),
    }),
  }
);

const tools = [
  readFile,
  writeFile,
  listDir,
  execCommand
]

// 绑定工具
const modelWithTools = model.bindTools(tools)

const messages = [
  new SystemMessage(`你是一个编码助手。
    当前平台：${process.platform}。
    命令在 ${process.platform === 'win32' ? 'cmd.exe' : '/bin/sh'} 中执行。
    必须使用对应平台的命令。
    启动开发服务器属于长期运行任务，不要通过等待退出的命令工具直接执行。`
  ),
  new HumanMessage('帮我创建一个vue项目并跑起来'),
]

let result = await modelWithTools.invoke(messages)
console.log(result)

while (result.tool_calls && result.tool_calls.length > 0) {
  messages.push(result)
  const toolCallResults = await Promise.all(result.tool_calls.map(async toolCall => {
    const toolArgs = toolCall.args;
    const toolName = toolCall.name;
    const tool = tools.find(tool => tool.name === toolName);
    if (!tool) {
      throw new Error(`Tool ${toolName} not found`);
    }
    console.log(`Tool call: ${toolName} with input: ${JSON.stringify(toolArgs)}`);
    const result = await tool.invoke(toolArgs);
    console.log(`Tool result: ${JSON.stringify(result)}`);
    return result;
  }));

  // 将工具结果添加到消息历史
  result.tool_calls.forEach((toolCall, index) => {
    messages.push(
      new ToolMessage({
        content: toolCallResults[index],
        tool_call_id: toolCall.id,
      })
    );
  });

  // 再次调用模型，传入工具结果
  result = await modelWithTools.invoke(messages);
}

console.log(result.content)