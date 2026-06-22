# Web 可视化管理系统

对应开发计划第 7 节。

## 结构
- `backend/app.py` —— FastAPI 后端，实现第7节全部接口。
- `frontend/index.html` —— 单页前端（实时画面 / 识别结果 / 机械臂控制 / 库房统计 / 操作日志）。

## 本机运行（演示用，mock 模式）

```bash
pip install fastapi uvicorn
cd web/backend
uvicorn app:app --reload --port 8000
```

浏览器打开 http://127.0.0.1:8000/ 即可看到完整页面（数据为 mock）。

## 接口一览

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/api/status` | GET | 系统状态（运行中/机械臂状态/是否 mock） |
| `/api/detections` | GET | 最近识别结果 |
| `/api/tasks/start` | POST | 开始自动识别抓取 |
| `/api/tasks/stop` | POST | 停止任务 |
| `/api/arm/home` | POST | 机械臂回到初始姿态 |
| `/api/inventory` | GET | 库房统计 + 最近日志 |
| `/api/inventory/reset` | POST | 清空统计 |
| `/video` | GET | 实时视频流（mock 占位） |

## 真机接入要点
1. `config/web.yaml` 把 `mock_mode` 改为 `false`。
2. `app_state.AppState._make_detector` 在 real 分支注入真实 detector
   （`build_detector("acl", "models/power_objects.om")`），并把 MockKinematics/
   MockArmDriver 换成 `Ros2Kinematics` / `SerialArmDriver`（见 power_arm_control）。
3. `ThrottledFrameSource` 传入 `real_reader`（OpenCV 摄像头读帧）。
4. `/video` 改为返回 MJPEG 流：cv2 抓帧 → 推理叠加检测框 → multipart 推送。

## 架构（阶段七）
- `app_state.py` 的 `AppState` 单例持有共享 `StatusReporter` + `Inventory` + runner 句柄。
- `/api/tasks/start` 在**后台线程**启动 `PowerSortingRunner`，与 FastAPI 主线程共享状态（加锁）。
- runner 的状态回调更新 `arm_status`；检测器被 `_CachingDetector` 包装，把每次识别写入共享状态。
- 各 GET 接口返回共享状态的只读快照；前端轮询展示。
