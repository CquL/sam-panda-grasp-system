#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import rospy
from std_msgs.msg import String


class TaskCommandCLI:
    def __init__(self):
        rospy.init_node("task_command_cli", anonymous=True)

        self.command_topic = rospy.get_param("~command_topic", "/vlm/task_command")
        self.target_topic = rospy.get_param("~target_topic", "/vlm/global_target")
        self.status_topic = rospy.get_param("~status_topic", "/demo/task_status")

        self.pub = rospy.Publisher(self.command_topic, String, queue_size=10)
        rospy.Subscriber(self.target_topic, String, self.target_cb, queue_size=10)
        rospy.Subscriber(self.status_topic, String, self.status_cb, queue_size=20)

        self.examples = [
            "抓一罐可乐和一瓶啤酒",
            "抓最右边那罐可乐、一瓶啤酒和一盒白砂糖",
            "抓一罐可乐、一瓶啤酒、一盒白砂糖和一个红色盒子",
        ]

        rospy.loginfo("任务输入终端已启动，发布到: %s", self.command_topic)

    def target_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            print("\n[VLM] 返回了无法解析的结果")
            return

        if not payload.get("ok", False):
            reason = payload.get("reason", "unknown")
            print(f"\n[VLM] 未找到目标，reason={reason}")
            return

        target = payload.get("target", "")
        confidence = payload.get("confidence", 0.0)
        entries = payload.get("entries", [])
        if entries:
            summary = " | ".join(
                [f"{item.get('target', '?')} {item.get('bbox', [])}" for item in entries]
            )
        else:
            boxes = payload.get("bboxes", [])
            summary = str(boxes)
        print(f"\n[VLM] 目标理解结果: target={target}, conf={confidence:.2f}, entries={summary}")

    def status_cb(self, msg):
        status = msg.data.strip()
        if status:
            print(f"\n[EXEC] 当前任务状态: {status}")

    def run(self):
        print("=" * 60)
        print("货架拣选任务输入终端")
        print("直接输入自然语言任务并回车即可发布到 /vlm/task_command")
        print("输入 exit / quit 退出")
        print("示例指令:")
        for i, example in enumerate(self.examples, start=1):
            print(f"  {i}. {example}")
        print("=" * 60)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                user_cmd = input("\n请输入任务指令: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出任务输入终端")
                break

            if not user_cmd:
                continue

            if user_cmd.lower() in {"exit", "quit", "q"}:
                print("退出任务输入终端")
                break

            self.pub.publish(String(data=user_cmd))
            print(f"[PUB] 已发布任务: {user_cmd}")

            # Give rospy callbacks time to print async feedback before next prompt.
            for _ in range(5):
                if rospy.is_shutdown():
                    break
                rate.sleep()


if __name__ == "__main__":
    try:
        cli = TaskCommandCLI()
        cli.run()
    except rospy.ROSInterruptException:
        pass
