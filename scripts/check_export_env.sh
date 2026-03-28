#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILURES=0
WARNINGS=0

pass() {
  printf '[PASS] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
  WARNINGS=$((WARNINGS + 1))
}

fail() {
  printf '[FAIL] %s\n' "$1"
  FAILURES=$((FAILURES + 1))
}

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "Command available: $1"
  else
    fail "Missing command: $1"
  fi
}

check_file() {
  if [ -f "$1" ]; then
    pass "File exists: $2"
  else
    fail "Missing file: $2"
  fi
}

check_dir() {
  if [ -d "$1" ]; then
    pass "Directory exists: $2"
  else
    fail "Missing directory: $2"
  fi
}

echo "== SAM Grasp System Export Environment Check =="
echo "Repository root: $ROOT_DIR"
echo

check_cmd python3
check_cmd git
check_cmd roscore
check_cmd roslaunch
check_cmd rospack
check_cmd catkin_make

echo
check_dir "$ROOT_DIR/src/sam_perception" "src/sam_perception"
check_dir "$ROOT_DIR/src/panda_moveit_config" "src/panda_moveit_config"
check_dir "$ROOT_DIR/src/panda_pick_place" "src/panda_pick_place"
check_dir "$ROOT_DIR/third_party/graspnet-baseline" "third_party/graspnet-baseline"

check_file "$ROOT_DIR/third_party/graspnet-baseline/checkpoint-rs.tar" "GraspNet checkpoint"

if [ -n "${SAM_CHECKPOINT_PATH:-}" ]; then
  check_file "$SAM_CHECKPOINT_PATH" "SAM checkpoint from SAM_CHECKPOINT_PATH"
elif [ -f "$ROOT_DIR/src/sam_perception/models/sam_vit_b_01ec64.pth" ]; then
  pass "SAM checkpoint exists in repository models directory"
else
  fail "SAM checkpoint missing. Set SAM_CHECKPOINT_PATH or add src/sam_perception/models/sam_vit_b_01ec64.pth"
fi

echo
if [ -n "${ANYGRASP_PYTHON:-}" ]; then
  if [ -x "$ANYGRASP_PYTHON" ]; then
    pass "ANYGRASP_PYTHON points to executable: $ANYGRASP_PYTHON"
  else
    fail "ANYGRASP_PYTHON is set but not executable: $ANYGRASP_PYTHON"
  fi
else
  warn "ANYGRASP_PYTHON not set. Launch files will fall back to /usr/bin/python3"
fi

if [ -n "${ROS_PYTHON_EXEC:-}" ]; then
  if [ -x "$ROS_PYTHON_EXEC" ]; then
    pass "ROS_PYTHON_EXEC points to executable: $ROS_PYTHON_EXEC"
  else
    fail "ROS_PYTHON_EXEC is set but not executable: $ROS_PYTHON_EXEC"
  fi
else
  warn "ROS_PYTHON_EXEC not set. Launch files will fall back to /usr/bin/python3"
fi

if [ -n "${DASHSCOPE_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
  pass "VLM API key environment detected"
else
  warn "No DASHSCOPE_API_KEY or OPENAI_API_KEY detected. VLM nodes will not run."
fi

if [ -n "${LIBFFI_PRELOAD:-}" ]; then
  if [ -f "$LIBFFI_PRELOAD" ]; then
    pass "LIBFFI_PRELOAD points to existing file"
  else
    warn "LIBFFI_PRELOAD is set but file does not exist: $LIBFFI_PRELOAD"
  fi
else
  warn "LIBFFI_PRELOAD not set. This is fine unless your environment specifically requires it."
fi

echo
if command -v rospack >/dev/null 2>&1; then
  for pkg in gazebo_ros franka_description franka_gazebo moveit_ros_move_group gazebo_ros_link_attacher; do
    if rospack find "$pkg" >/dev/null 2>&1; then
      pass "ROS package found: $pkg"
    else
      warn "ROS package not found in current environment: $pkg"
    fi
  done
fi

echo
if grep -R "/home/lhj/.gazebo/models/Cracker_Box/textured.obj" "$ROOT_DIR/src/panda_pick_place/worlds" >/dev/null 2>&1; then
  warn "World files still reference a machine-local Cracker_Box mesh. Portability is not yet complete."
fi

echo
if [ "$FAILURES" -gt 0 ]; then
  echo "Result: FAIL ($FAILURES failures, $WARNINGS warnings)"
  exit 1
fi

echo "Result: PASS with $WARNINGS warnings"
