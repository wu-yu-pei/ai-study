import dotenv from "dotenv";
import { createAgent, tool } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { z } from "zod";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(scriptDir, ".env"), quiet: true });

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
});

const writeFile = tool(({ filePath, fileContent }) => {
    filePath = path.resolve(scriptDir, filePath);
    fs.writeFileSync(filePath, fileContent, 'utf-8');
    return 'success';
}, {
    name: "writeFile",
    description: "Write the content of a file",
    schema: z.object({
        filePath: z.string().describe("The path to the file"),
        fileContent: z.string().describe("The content of the file"),
    }),
})


const agent = createAgent({
    model: new ChatOpenAI({
        model: process.env.MODEL_NAME,
        apiKey: process.env.OPENAI_API_KEY,
        configuration: {
            baseURL: process.env.OPENAI_BASE_URL,
        },
    }),
    tools: [getFileContent, writeFile],
});

const result = await agent.invoke({
    messages: [
        {
            role: "user",
            content: "What is the file package.json content?, ./package.json , 把这个content的内容写到 ./package2.json",
        }
    ]
})

console.log(result.messages.at(-1).content);
