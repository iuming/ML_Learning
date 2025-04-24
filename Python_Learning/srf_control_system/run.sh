#!/bin/bash
# 启动仿真器进程
python -m simulator.rf_simulator &
SIM_PID=$!

# 启动控制程序进程
python -m controller.control_loop &
CTRL_PID=$!

# 等待进程结束
wait $SIM_PID $CTRL_PID

# 清理资源（需实现信号处理）
kill $SIM_PID $CTRL_PID