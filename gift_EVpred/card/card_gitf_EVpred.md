【任务定义】
- 对每个 click 事件（用户进入直播间时刻）：
  - 输入：((u, s, live, t_click)) + 进入时刻可用特征
  - 输出：未来窗口 [t_click, t_click+W] 内礼物金额总和（或其期望），即 EV
  - 标签：gift_price_label = sum(gift_price) (窗口内)；可用 target=log1p(label)，并有 is_gift 辅助


下一步：
     A. Frozen：仅用 Train 窗口内历史计算 pair/user/streamer 的统计量；Val/Test 只查表
     B. Rolling：对每个 click 严格用 gift_ts < click_ts（searchsorted(side='left')）构造 past-only 特征
