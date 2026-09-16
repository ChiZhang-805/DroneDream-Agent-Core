import { Bot } from "lucide-react";

import ai21 from "@lobehub/icons-static-svg/icons/ai21-brand-color.svg";
import azureAi from "@lobehub/icons-static-svg/icons/azureai-color.svg";
import baichuan from "@lobehub/icons-static-svg/icons/baichuan-color.svg";
import bedrock from "@lobehub/icons-static-svg/icons/bedrock-color.svg";
import cerebras from "@lobehub/icons-static-svg/icons/cerebras-brand-color.svg";
import claude from "@lobehub/icons-static-svg/icons/claude-color.svg";
import claudeCode from "@lobehub/icons-static-svg/icons/claudecode-color.svg";
import cloudflare from "@lobehub/icons-static-svg/icons/cloudflare-color.svg";
import codex from "@lobehub/icons-static-svg/icons/codex-color.svg";
import cohere from "@lobehub/icons-static-svg/icons/cohere-color.svg";
import deepseek from "@lobehub/icons-static-svg/icons/deepseek-color.svg";
import doubao from "@lobehub/icons-static-svg/icons/doubao-color.svg";
import fireworks from "@lobehub/icons-static-svg/icons/fireworks-color.svg";
import gemini from "@lobehub/icons-static-svg/icons/gemini-color.svg";
import geminiCli from "@lobehub/icons-static-svg/icons/geminicli-color.svg";
import github from "@lobehub/icons-static-svg/icons/github.svg";
import githubCopilot from "@lobehub/icons-static-svg/icons/githubcopilot.svg";
import grok from "@lobehub/icons-static-svg/icons/grok.svg";
import groq from "@lobehub/icons-static-svg/icons/groq.svg";
import huggingFace from "@lobehub/icons-static-svg/icons/huggingface-color.svg";
import hunyuan from "@lobehub/icons-static-svg/icons/hunyuan-color.svg";
import lmStudio from "@lobehub/icons-static-svg/icons/lmstudio.svg";
import minimax from "@lobehub/icons-static-svg/icons/minimax-color.svg";
import mistral from "@lobehub/icons-static-svg/icons/mistral-color.svg";
import modelscope from "@lobehub/icons-static-svg/icons/modelscope-color.svg";
import novita from "@lobehub/icons-static-svg/icons/novita-color.svg";
import nvidia from "@lobehub/icons-static-svg/icons/nvidia-color.svg";
import ollama from "@lobehub/icons-static-svg/icons/ollama.svg";
import openai from "@lobehub/icons-static-svg/icons/openai.svg";
import openrouter from "@lobehub/icons-static-svg/icons/openrouter-color.svg";
import perplexity from "@lobehub/icons-static-svg/icons/perplexity-color.svg";
import qwen from "@lobehub/icons-static-svg/icons/qwen-color.svg";
import replicate from "@lobehub/icons-static-svg/icons/replicate.svg";
import sambanova from "@lobehub/icons-static-svg/icons/sambanova-color.svg";
import siliconCloud from "@lobehub/icons-static-svg/icons/siliconcloud-color.svg";
import stepfun from "@lobehub/icons-static-svg/icons/stepfun-color.svg";
import together from "@lobehub/icons-static-svg/icons/together-color.svg";
import upstage from "@lobehub/icons-static-svg/icons/upstage-color.svg";
import vertexAi from "@lobehub/icons-static-svg/icons/vertexai-color.svg";
import xai from "@lobehub/icons-static-svg/icons/xai.svg";
import yi from "@lobehub/icons-static-svg/icons/yi-color.svg";
import zhipu from "@lobehub/icons-static-svg/icons/zhipu-color.svg";

export const PROVIDER_ICON_ASSET_COUNT = 43;

const KIMI_PATH = "M21.765.351C22.998.351 24 1.353 24 2.586S22.998 4.82 21.765 4.82h-1.974c-.15 0-.26-.12-.26-.26V2.586A2.237 2.237 0 0 1 21.765.35M9.41 13.388l8.447-8.377c.16-.16.07-.471-.14-.471h-4.55s-.1.02-.14.06l-9.099 9.029c-.14.14-.35.02-.35-.21V4.81c0-.15-.1-.27-.221-.27H.22c-.12 0-.22.12-.22.27v18.57c0 .15.1.27.22.27h3.137c.12 0 .22-.12.22-.27v-3.79c0-.08.03-.16.08-.21l2.826-2.796c.07-.07.16-.08.241-.03l7.546 5.551a8.9 8.9 0 004.018 1.493c.12.01.23-.11.23-.27V19.76c0-.14-.08-.25-.19-.26a5.8 5.8 0 01-2.355-.942l-6.533-4.73c-.14-.09-.15-.32-.03-.441";

const PROVIDER_LOGOS: Record<string, string> = {
  ai21,
  anthropic: claude,
  azure: azureAi,
  "azure-ai": azureAi,
  "azure-openai": azureAi,
  baichuan,
  bedrock,
  cerebras,
  chatglm: zhipu,
  claude,
  claudecode: claudeCode,
  cloudflare,
  codex,
  cohere,
  deepseek,
  doubao,
  fireworks,
  gemini,
  geminicli: geminiCli,
  github,
  "github-models": github,
  githubcopilot: githubCopilot,
  glm: zhipu,
  google: gemini,
  grok,
  groq,
  huggingface: huggingFace,
  hunyuan,
  lmstudio: lmStudio,
  minimax,
  mistral,
  modelscope,
  novita,
  nvidia,
  ollama,
  openai,
  openrouter,
  perplexity,
  qwen,
  replicate,
  sambanova,
  siliconcloud: siliconCloud,
  siliconflow: siliconCloud,
  stepfun,
  together,
  upstage,
  vertexai: vertexAi,
  xai,
  yi,
  zhipu,
};

export function ProviderLogo({ provider, icon }: { provider: string; icon?: string }) {
  const key = (icon || provider).trim().toLowerCase();
  const providerKey = provider.trim().toLowerCase();
  if (key === "kimi" || key === "moonshot" || providerKey === "kimi" || providerKey === "moonshot") {
    return <svg className="provider-logo provider-logo-kimi" viewBox="0 0 24 24" aria-hidden="true"><path d={KIMI_PATH} /></svg>;
  }
  const source = PROVIDER_LOGOS[key] ?? PROVIDER_LOGOS[providerKey];
  if (source) return <img className="provider-logo" src={source} alt="" aria-hidden="true" />;
  return <span className="provider-logo provider-logo-generic" aria-hidden="true"><Bot /></span>;
}
