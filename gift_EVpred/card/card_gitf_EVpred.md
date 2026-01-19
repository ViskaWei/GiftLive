【任务定义】
- 对每个 click 事件（用户进入直播间时刻）：
  - 输入：((u, s, live, t_click)) + 进入时刻可用特征
  - 输出：未来窗口 [t_click, t_click+W] 内礼物金额总和（或其期望），即 EV
  - 标签：gift_price_label = sum(gift_price) (窗口内)；可用 target=log1p(label)，并有 is_gift 辅助


下一步：
     A. Frozen：仅用 Train 窗口内历史计算 pair/user/streamer 的统计量；Val/Test 只查表
     B. Rolling：对每个 click 严格用 gift_ts < click_ts（searchsorted(side='left')）构造 past-only 特征

1-7每天打赏是不一样的需要单独建模

2.1 先定“估计层”的正确预测目标：不是预测 gift，而是预测“行动的后果”
你写的长期目标是：

Total Revenue + 用户留存/满意度 + 主播生态健康

这不是单一 reward。一个可落地的分解是：对每次“把某用户分配/曝光给某主播”的行动 a，要预测：

短期收益
( r^{rev}{t} = \mathbb{E}[\text{gift_amount}{t:t+H} \mid u,s,ctx,a] )
用户侧长期（留存/满意度的代理指标）
( r^{usr}{t} = \mathbb{E}[\text{return}{t+1d} \mid u,history,a] )
或 ( \mathbb{E}[\Delta \text{watch_time}, \Delta \text{engagement} \mid a] )
主播生态健康（外部性/约束项）
例如：主播侧 exposure / revenue 的集中度、长尾扶持程度、过载惩罚
( r^{eco}_{t} = -\lambda \cdot \text{concentration}(\text{exposure/revenue}) )
或者每个主播一个 concave utility：( U_s(x) ) 边际递减（避免“让某个主播爆”）
关键点：你要预测的是“给了这个行动，会发生什么”，而不是“这条样本是否送礼”。