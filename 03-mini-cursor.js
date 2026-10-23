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

// 仅裁剪控制台预览，传给模型的工具结果始终保留完整内容。
function preview(value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const lines = String(text ?? '').replace(/\r\n/g, '\n').split('\n');
  const visible = lines.slice(0, 20).join('\n');
  const clipped = visible.slice(0, 2000);
  return clipped + (lines.length > 20 || visible.length > 2000
    ? '\n…（控制台预览已截断）' : '');
}

function logBlock(title, content) {
  console.log(`\n── ${title} ──\n${content}`);
}

function formatToolResult(name, value) {
  if (name === 'execCommand') {
    const { exitCode, stdout, stderr } = typeof value === 'string'
      ? JSON.parse(value) : value;
    return [
      `退出码：${exitCode ?? '未正常退出'}`,
      stdout?.trim() ? `标准输出：\n${preview(stdout.trimEnd())}` : '标准输出：（空）',
      stderr?.trim() ? `标准错误：\n${preview(stderr.trimEnd())}` : '',
    ].filter(Boolean).join('\n');
  }
  if (name === 'listDir' && Array.isArray(value)) {
    return `共 ${value.length} 项\n${preview(value.join('\n'))}`;
  }
  return preview(value);
}

function messageText(content) {
  return typeof content === 'string' ? content : content
    .filter(block => block.type === 'text')
    .map(block => block.text).join('\n');
}

let round = 0;
async function requestModel() {
  const currentRound = ++round;
  const started = Date.now();
  console.log(`\n[第 ${currentRound} 轮] 正在等待模型响应…`);
  const timer = setInterval(() => {
    console.log(`[第 ${currentRound} 轮] 仍在等待（${Math.round((Date.now() - started) / 1000)} 秒）…`);
  }, 10000);
  try {
    const response = await modelWithTools.invoke(messages);
    console.log(`[第 ${currentRound} 轮] 模型已返回，用时 ${((Date.now() - started) / 1000).toFixed(1)} 秒，工具调用 ${response.tool_calls?.length ?? 0} 个`);
    return response;
  } catch (error) {
    console.error(`[第 ${currentRound} 轮] 模型请求失败：${error.message}`);
    throw error;
  } finally {
    clearInterval(timer);
  }
}

let result = await requestModel();

while (result.tool_calls && result.tool_calls.length > 0) {
  messages.push(result)
  const explanation = messageText(result.content);
  if (explanation) logBlock('模型说明', explanation);
  const toolCallResults = await Promise.all(result.tool_calls.map(async (toolCall, index) => {
    const toolArgs = toolCall.args;
    const toolName = toolCall.name;
    const tool = tools.find(tool => tool.name === toolName);
    if (!tool) {
      throw new Error(`Tool ${toolName} not found`);
    }
    const label = `${round}.${index + 1} ${toolName}`;
    logBlock(`工具 ${label} · 开始`, preview(toolArgs));
    const started = Date.now();
    try {
      const result = await tool.invoke(toolArgs);
      logBlock(`工具 ${label} · 返回 · ${((Date.now() - started) / 1000).toFixed(1)} 秒`,
        formatToolResult(toolName, result));
      return result;
    } catch (error) {
      logBlock(`工具 ${label} · 失败`, error.message);
      throw error;
    }
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
  result = await requestModel();
}

logBlock('最终回答', messageText(result.content));
