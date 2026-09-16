import type { TaskThread } from "./types";

export interface PlanApproval {
  messageId: string;
  revisionId: string;
}

type ApprovalThread = Pick<TaskThread, "state" | "archived" | "messages">;

// 功能：
//   1. 只为等待确认任务中的最新计划提供确认身份，历史卡片不能授权另一份计划。
//   2. 缺失或歧义身份直接拒绝；服务端仍须重新核验计划、资产及一次性执行权限。
// 输入：
//   thread：当前界面的任务状态和按消息顺序排列的记录，未选任务时为 undefined。
// 输出：
//   approval：允许确认的消息与计划身份；当前不可确认时为 null。
export function currentPlanApproval(thread: ApprovalThread | undefined): PlanApproval | null {
  if (!thread || thread.archived || thread.state !== "awaiting_confirmation") return null;
  const messages = thread.messages ?? [];
  const plan = [...messages].reverse().find((message) => message.kind === "plan");
  const revision = plan?.metadata?.plan_revision_id;
  if (!plan || typeof plan.message_id !== "string" || !plan.message_id.trim()
    || typeof revision !== "string" || !revision.trim() || revision !== revision.trim()) return null;
  if (messages.filter((message) => message.message_id === plan.message_id).length !== 1) return null;
  const approval = { messageId: plan.message_id, revisionId: revision };
  return approval;
}
