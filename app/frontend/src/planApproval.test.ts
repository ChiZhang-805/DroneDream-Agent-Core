import { expect, it } from "vitest";

import { currentPlanApproval } from "./planApproval";
import type { Message } from "./types";

// 功能：
//   为确认身份测试建立独立计划消息，不共享可变 metadata。
// 输入：
//   id：用户实际看到的消息身份。
//   revision：计划对应的修订身份，允许注入非法测试值。
// 输出：
//   message：仅供本文件测试的计划消息。
function plan(id: string, revision: unknown): Message {
  const message: Message = {
    message_id: id, sequence: 0, role: "assistant", kind: "plan", content: "test plan",
    metadata: { plan_revision_id: revision }, created_at: "2026-09-16T00:00:00Z",
  };
  return message;
}

// 功能：
//   确认结果必须同时标识最新卡片与其计划，不能仅返回一个可由旧卡片借用的计划值。
// 输入：
//   无。
// 输出：
//   None：断言最新卡片与修订身份一致，旧卡片身份不获准。
it("binds approval to the exact latest message, not a historical card", () => {
  const approval = currentPlanApproval({ state: "awaiting_confirmation", archived: false,
    messages: [plan("old-card", "old-plan"), plan("new-card", "new-plan")] });
  expect(approval).toEqual({ messageId: "new-card", revisionId: "new-plan" });
  expect(approval?.messageId).not.toBe("old-card");
});

// 功能：
//   非等待确认状态的旧计划不得在界面重复提供启动许可。
// 输入：
//   state：执行、结束及未准备完成的各任务状态。
// 输出：
//   None：所有这些状态均无可确认计划。
it.each(["created", "preparing", "executing", "holding", "landing", "completed", "failed", "cancelled"])(
  "does not enable execution in state %s", (state) => {
    expect(currentPlanApproval({ state, archived: false, messages: [plan("card", "plan")] }))
      .toBeNull();
  },
);

// 功能：
//   归档、未选任务或尚无消息时不能产生空壳执行授权。
// 输入：
//   无。
// 输出：
//   None：三种情况均返回 null。
it("rejects archived, absent and empty threads", () => {
  expect(currentPlanApproval(undefined)).toBeNull();
  expect(currentPlanApproval({ state: "awaiting_confirmation", archived: false })).toBeNull();
  expect(currentPlanApproval({ state: "awaiting_confirmation", archived: true,
    messages: [plan("card", "plan")] })).toBeNull();
});

// 功能：
//   最新计划损坏时拒绝确认，不回退到更早且看似合法的计划。
// 输入：
//   revision：空值、错误类型或含边缘空白的修订字段。
// 输出：
//   None：不生成可确认计划。
it.each([undefined, null, "", " ", " plan", 5, false])(
  "does not fall back past invalid latest revision %s", (revision) => {
    expect(currentPlanApproval({ state: "awaiting_confirmation", archived: false,
      messages: [plan("older", "valid"), plan("latest", revision)] })).toBeNull();
  },
);

// 功能：
//   重复卡片身份会使界面点击来源歧义，必须拒绝而不是按数组位置猜测。
// 输入：
//   无。
// 输出：
//   None：重复消息不能获得确认身份。
it("rejects duplicate message identities", () => {
  expect(currentPlanApproval({ state: "awaiting_confirmation", archived: false,
    messages: [plan("same", "older"), plan("same", "newer")] })).toBeNull();
});

// 功能：
//   普通状态消息不取代最新计划，读取也不能反转调用方的原消息数组。
// 输入：
//   无。
// 输出：
//   None：计划身份正确，输入顺序与对象保持不变。
it("ignores later status messages without mutating message order", () => {
  const messages = [plan("first", "first-plan"), plan("last", "last-plan"),
    { ...plan("status", null), kind: "status" as const }];
  const original = [...messages];
  const approval = currentPlanApproval({ state: "awaiting_confirmation", archived: false, messages });
  expect(approval).toEqual({ messageId: "last", revisionId: "last-plan" });
  expect(messages).toEqual(original);
});
