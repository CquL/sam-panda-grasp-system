#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import time

import rospy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32, String


class ExperimentLogger:
    FIELDNAMES = [
        "run_index",
        "scene_label",
        "notes",
        "wall_start",
        "start_sim_time",
        "end_sim_time",
        "targets_dispatched",
        "done_count",
        "failed_count",
        "failed_by_user_count",
        "gsr",
        "tct_sec",
        "cartesian_success_count",
        "cartesian_total_count",
        "psr_threshold",
        "psr",
        "joint_cost_rad",
        "scheduler_time_sec",
        "cluster_count",
        "candidate_pose_count",
    ]

    def __init__(self):
        rospy.init_node("experiment_logger")

        self.scene_label = rospy.get_param("~scene_label", "unknown")
        self.notes = rospy.get_param("~notes", "")
        self.output_file = os.path.expanduser(
            rospy.get_param(
                "~output_file",
                "~/grasp_robot_ws/experiment_results/end_to_end_metrics.csv",
            )
        )
        self.psr_threshold = float(rospy.get_param("~psr_threshold", 0.95))
        self.finalize_delay = float(rospy.get_param("~finalize_delay", 0.2))

        self.run_index = 0
        self.current_run = None
        self.finalize_timer = None
        self.pending_scheduler_metrics = None

        self.ensure_output_file()

        rospy.Subscriber("/graspnet/grasp_pose_array", PoseArray, self.dispatch_cb, queue_size=10)
        rospy.Subscriber("/demo/task_status", String, self.status_cb, queue_size=50)
        rospy.Subscriber("/demo/command", String, self.command_cb, queue_size=10)
        rospy.Subscriber("/demo/cartesian_plan_fraction", Float32, self.fraction_cb, queue_size=200)
        rospy.Subscriber("/scheduler/run_metrics", String, self.scheduler_metrics_cb, queue_size=20)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            f"🧾 实验记录器已启动，数据将写入: {self.output_file} "
            f"(scene_label={self.scene_label}, psr_threshold={self.psr_threshold:.2f})"
        )

    def ensure_output_file(self):
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(self.output_file) or os.path.getsize(self.output_file) == 0:
            with open(self.output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def new_run(self):
        self.run_index += 1
        sim_now = rospy.Time.now().to_sec()
        self.current_run = {
            "run_index": self.run_index,
            "scene_label": self.scene_label,
            "notes": self.notes,
            "wall_start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_sim_time": sim_now,
            "end_sim_time": sim_now,
            "targets_dispatched": 0,
            "done_count": 0,
            "failed_count": 0,
            "failed_by_user_count": 0,
            "cartesian_success_count": 0,
            "cartesian_total_count": 0,
            "joint_cost_rad": "",
            "scheduler_time_sec": "",
            "cluster_count": "",
            "candidate_pose_count": "",
        }
        if self.pending_scheduler_metrics:
            self.current_run.update(self.pending_scheduler_metrics)
            self.pending_scheduler_metrics = None

    def cancel_finalize_timer(self):
        if self.finalize_timer is not None:
            self.finalize_timer.shutdown()
            self.finalize_timer = None

    def dispatch_cb(self, msg):
        if len(msg.poses) == 0:
            return

        self.cancel_finalize_timer()

        if self.current_run is None:
            self.new_run()
            rospy.loginfo(
                f"📝 开始记录第 {self.current_run['run_index']} 轮实验 "
                f"(scene={self.scene_label})"
            )

        self.current_run["targets_dispatched"] += len(msg.poses)

    def status_cb(self, msg):
        if self.current_run is None:
            return

        status = msg.data.strip()
        self.current_run["end_sim_time"] = rospy.Time.now().to_sec()

        if status == "DONE":
            self.current_run["done_count"] += 1
        elif status == "FAILED":
            self.current_run["failed_count"] += 1
        elif status == "FAILED_BY_USER":
            self.current_run["failed_by_user_count"] += 1

    def scheduler_metrics_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"⚠️ 解析 /scheduler/run_metrics 失败: {e}")
            return

        metrics = {
            "joint_cost_rad": payload.get("joint_cost_rad", ""),
            "scheduler_time_sec": payload.get("scheduler_time_sec", ""),
            "cluster_count": payload.get("cluster_count", ""),
            "candidate_pose_count": payload.get("candidate_pose_count", ""),
        }

        if self.current_run is None:
            self.pending_scheduler_metrics = metrics
        else:
            self.current_run.update(metrics)

    def fraction_cb(self, msg):
        if self.current_run is None:
            return

        fraction = float(msg.data)
        self.current_run["cartesian_total_count"] += 1
        if fraction >= self.psr_threshold:
            self.current_run["cartesian_success_count"] += 1

    def command_cb(self, msg):
        if self.current_run is None:
            return

        command = msg.data.strip().lower()
        if command == "all_done":
            self.cancel_finalize_timer()
            self.finalize_timer = rospy.Timer(
                rospy.Duration(self.finalize_delay),
                self.finalize_timer_cb,
                oneshot=True,
            )

    def finalize_timer_cb(self, _event):
        self.finalize_run()

    def finalize_run(self):
        if self.current_run is None:
            return

        self.cancel_finalize_timer()

        run = dict(self.current_run)
        total_targets = int(run["targets_dispatched"])
        done_count = int(run["done_count"])
        failed_count = int(run["failed_count"])
        failed_by_user_count = int(run["failed_by_user_count"])
        cartesian_success_count = int(run["cartesian_success_count"])
        cartesian_total_count = int(run["cartesian_total_count"])

        gsr = (done_count / total_targets) if total_targets > 0 else 0.0
        tct_sec = max(0.0, float(run["end_sim_time"]) - float(run["start_sim_time"]))
        psr = (
            cartesian_success_count / cartesian_total_count
            if cartesian_total_count > 0
            else 0.0
        )

        row = {
            "run_index": run["run_index"],
            "scene_label": run["scene_label"],
            "notes": run["notes"],
            "wall_start": run["wall_start"],
            "start_sim_time": f"{run['start_sim_time']:.3f}",
            "end_sim_time": f"{run['end_sim_time']:.3f}",
            "targets_dispatched": total_targets,
            "done_count": done_count,
            "failed_count": failed_count,
            "failed_by_user_count": failed_by_user_count,
            "gsr": f"{gsr:.4f}",
            "tct_sec": f"{tct_sec:.3f}",
            "cartesian_success_count": cartesian_success_count,
            "cartesian_total_count": cartesian_total_count,
            "psr_threshold": f"{self.psr_threshold:.2f}",
            "psr": f"{psr:.4f}",
            "joint_cost_rad": (
                f"{float(run['joint_cost_rad']):.4f}" if run["joint_cost_rad"] != "" else ""
            ),
            "scheduler_time_sec": (
                f"{float(run['scheduler_time_sec']):.4f}" if run["scheduler_time_sec"] != "" else ""
            ),
            "cluster_count": run["cluster_count"],
            "candidate_pose_count": run["candidate_pose_count"],
        }

        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

        rospy.loginfo(
            f"✅ 第 {run['run_index']} 轮实验已写入文件: "
            f"GSR={gsr:.2%}, TCT={tct_sec:.2f}s, PSR={psr:.2%}, "
            f"targets={total_targets}, done={done_count}, failed={failed_count}, "
            f"joint_cost={row['joint_cost_rad']}"
        )

        self.current_run = None

    def on_shutdown(self):
        # 若节点在 all_done 前被关闭，尽量把当前轮数据保存下来，避免白跑。
        if self.current_run is not None and self.current_run["targets_dispatched"] > 0:
            rospy.logwarn("⚠️ 实验记录器关闭时检测到未完成轮次，尝试保存当前统计。")
            self.current_run["end_sim_time"] = rospy.Time.now().to_sec()
            self.finalize_run()


if __name__ == "__main__":
    try:
        ExperimentLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
