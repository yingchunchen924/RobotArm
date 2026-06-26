#!/usr/bin/env python3
"""DOFBOT 垃圾分类抓取-放置 (自适应底边法定位).

核心修正(2026-06-24): 用方块贴地底边映射 y, 而非顶面质心,
避免顶面投影偏远导致 joint2 趋0 夹空. 全程 j5=265 夹爪不翻转.

用法:
  python3 sort_grab.py detect          # 只检测打印, 不动
  python3 sort_grab.py grab            # 抓取并放到对应颜色区
  python3 sort_grab.py grab --color red
"""
from __future__ import annotations
import argparse, sys, time
import cv2, numpy as np

CFG = "/home/HwHiAiUser/RobotArm/config"
GRAP = 175          # 夹紧角(方块尺寸)
J5 = 265            # 夹爪朝向, 全程固定不翻转
OPEN = 0
RELEASE = 30

# 颜色 HSV (red 跨 0/180 两段). 远端方块暗/小, S/V 下限放宽到 60 防漏检.
COLORS = {
    "red":    [((0,90,60),(8,255,255)), ((170,90,60),(180,255,255))],
    "yellow": [((22,80,80),(35,255,255))],
    "green":  [((45,60,60),(75,255,255))],
    "blue":   [((105,80,60),(125,255,255))],
}

# 方块颜色 -> 放置区接触姿态 [j1,j2,j3,j4] (j5=265,夹爪松开). 红区已标定; 其余待标.
PLACE = {
    "red":    [47, 18, 90, 12],     # 有害垃圾, 已验证
    "yellow": [133, 18, 90, 12],    # 其他垃圾 -> 黑区(左远), 红区镜像, 已标定
    "green":  [149, 56, 20, 34],    # 厨余 -> 绿区(左近), 蓝区镜像+伸长, 已标定
    "blue":   [31, 67, 4, 41],      # 可回收 -> 蓝区(右近), 已标定
}
# 放置区上方过渡姿态 (高位, j2大) 按 j1 区分方向
PLACE_ABOVE = {
    "red":    [47, 60, 80, 30],
    "yellow": [133, 60, 80, 30],
    "green":  [149, 80, 20, 34],
    "blue":   [31, 80, 4, 41],
}


def load_affine():
    A = []
    for line in open(CFG + "/affine.txt"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = [float(v) for v in s.replace(",", " ").split()]
        if len(p) >= 3:
            A.append(p[:3])
    fy = open(CFG + "/offset.txt").read().split()
    yoff, xoff = float(fy[0]), float(fy[1])
    return A, xoff, yoff


def px2xy(A, xoff, yoff, cx, cy):
    x = A[0][0]*cx + A[0][1]*cy + A[0][2] + xoff
    y = A[1][0]*cx + A[1][1]*cy + A[1][2] + yoff
    return round(x, 4), round(y, 4)


def classify_hue(hsv, c):
    """取轮廓内像素, 投票决定主色(防止蓝方块零星红像素被误判). 返回主色名或 None."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), 1)   # 收缩到内部, 避边缘
    votes = {}
    for nm, rngs in COLORS.items():
        cm = None
        for lo, hi in rngs:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi))
            cm = mm if cm is None else cv2.bitwise_or(cm, mm)
        votes[nm] = int(cv2.countNonZero(cv2.bitwise_and(cm, mask)))
    total = int(cv2.countNonZero(mask))
    if total == 0:
        return None
    top = max(votes, key=votes.get)
    # 主色必须占轮廓内 >40% 才认; 否则判定不可靠
    return top if votes[top] / total > 0.40 else None


def detect_block(frame, color):
    """返回 (color_name, gcx, gcy_ground) 占地中心像素, 或 None.
    用主色投票确定颜色, 避免红/蓝误判."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    names = [color] if color != "any" else list(COLORS)
    best = None
    for nm in names:
        m = None
        for lo, hi in COLORS[nm]:
            mm = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = mm if m is None else cv2.bitwise_or(m, mm)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts = [c for c in cv2.findContours(m, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 1000]
        for c in cnts:
            # 主色验证: 该轮廓的主导色必须就是 nm, 否则跳过(防误判)
            if classify_hue(hsv, c) != nm:
                continue
            a = cv2.contourArea(c)
            if best is None or a > best[0]:
                best = (a, nm, c)
    if best is None:
        return None
    _, nm, c = best
    M = cv2.moments(c)
    gcx = int(M["m10"]/M["m00"])
    x, y, w, h = cv2.boundingRect(c)
    # 贴地底边: 仅在【本轮廓内】沿中心列向下找方块底沿, 不能扫到下方其它方块.
    # 从顶面mask的bbox下沿(y+h)开始, 向下找仍属于同一方块(非白垫)的最低行,
    # 限制最多延伸 1.2*h (方块侧面投影高度量级), 避免串到下方方块.
    x0, x1 = max(0, gcx-12), min(W, gcx+12)
    rowmin = gray[:, x0:x1].min(axis=1)
    limit = min(H-1, y + int(h*2.2))      # 本方块底沿不会超过顶面顶部往下 2.2*h
    bb = y + h
    for yy in range(y+h, limit):
        if rowmin[yy] < 110:              # 仍是方块(印字/边/阴影)
            bb = yy
        elif yy - bb > 12:                # 连续12行都是白垫 -> 方块结束
            break
    gcy = bb - 8        # 往占地中心挪一点点
    return nm, gcx, gcy


class IK:
    def __init__(self):
        import rclpy
        from dofbot_info.srv import Kinemarics
        self.rclpy = rclpy
        self.K = Kinemarics
        rclpy.init()
        self.node = rclpy.create_node("sort_grab")
        self.cli = self.node.create_client(Kinemarics, "trial_service")
        if not self.cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("trial_service 不可用, 先启动 IK 服务")

    def solve(self, x, y):
        req = self.K.Request()
        req.tar_x, req.tar_y, req.tar_z = float(x), float(y), 0.0
        req.kin_name = "ik"
        fut = self.cli.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, fut)
        r = fut.result()
        j = [r.joint1, r.joint2, r.joint3, r.joint4, r.joint5]
        if j[2] < 0:                      # 官方负值修正(判 joint3)
            j[1] += j[2]/2
            j[3] += j[2]*3/4
            j[2] = 0
        return [round(v, 2) for v in j]

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def make_arm():
    import Arm_Lib
    return Arm_Lib.Arm_Device()


def w6(arm, a, ms):
    arm.Arm_serial_servo_write6_array(list(a), ms)
    time.sleep(ms/1000.0 + 0.2)


def w1(arm, i, a, ms):
    arm.Arm_serial_servo_write(i, a, ms)
    time.sleep(ms/1000.0 + 0.2)


def open_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, 640); cap.set(4, 480)
    return cap


def capture(arm):
    """回拍照位, 等机械臂运动停稳后再取帧.
    上一个方块放完甩回拍照位时电机仍在抖, 立即取帧会糊/暗导致远端方块漏检,
    故回位后额外静候 1.2s, 并多丢帧、取最后一帧."""
    w6(arm, [91, 135, 0, 0, 90, RELEASE], 1500)
    time.sleep(1.2)                       # 等画面彻底稳定再开相机
    cap = open_cam()
    f = None
    for _ in range(30):                   # 多丢帧, 等曝光/对焦收敛
        ok, f = cap.read(); time.sleep(0.05)
    cap.release()
    return f


def cmd_detect(args):
    A, xoff, yoff = load_affine()
    arm = make_arm()
    f = capture(arm)
    d = detect_block(f, args.color)
    if not d:
        print("NO BLOCK"); return 0
    nm, gcx, gcy = d
    x, y = px2xy(A, xoff, yoff, gcx, gcy)
    print("COLOR=%s ground_px=(%d,%d) -> IK=(%.4f,%.4f)" % (nm, gcx, gcy, x, y))
    return 0


def grab_one(arm, A, xoff, yoff, ik, color):
    """拍照->检测 color->抓取->放到对应区. 复用传入的 arm 与 IK 连接.
    返回 True 成功放置, False 未检测到方块."""
    f = capture(arm)
    d = detect_block(f, color)
    if not d:
        print("NO BLOCK (%s)" % color); return False
    nm, gcx, gcy = d
    x, y = px2xy(A, xoff, yoff, gcx, gcy)
    print("COLOR=%s -> IK=(%.4f,%.4f)" % (nm, x, y))
    joints = ik.solve(x, y)
    print("GRASP joints=%s j2=%.1f" % (joints, joints[1]))
    if joints[1] < 3:
        print("WARN joint2=%.1f 过低, 方块可能太远(y>0.33)夹空风险" % joints[1])
    # 抓取(官方序列, j5=265)
    w6(arm, [90, 80, 50, 50, J5, GRAP], 1000)
    w1(arm, 6, OPEN, 500)
    w6(arm, [joints[0], joints[1], joints[2], joints[3], J5, OPEN], 1000)
    w1(arm, 6, GRAP, 800); time.sleep(0.6)
    w6(arm, [joints[0], 80, 50, 50, J5, GRAP], 1200)   # 抬起
    # 放置
    place = PLACE.get(nm)
    above = PLACE_ABOVE.get(nm)
    if place is None:
        print("放置区 %s 未标定, 方块抓起停在上方. 请标定 PLACE[%s]" % (nm, nm))
        return True
    if above:
        w6(arm, above + [J5, GRAP], 1300)
    # 分步降到放置点
    w6(arm, [place[0], place[1]+22, place[2], place[3]-8, J5, GRAP], 1000)
    w6(arm, [place[0], place[1]+10, place[2], place[3]-3, J5, GRAP], 1000)
    w6(arm, place + [J5, GRAP], 1200)
    w1(arm, 6, RELEASE, 700); time.sleep(0.5)
    if above:
        w6(arm, above + [J5, RELEASE], 1200)
    w6(arm, [91, 135, 0, 0, 90, RELEASE], 1500)
    print("DONE -> %s placed" % nm)
    return True


def cmd_grab(args):
    A, xoff, yoff = load_affine()
    arm = make_arm()
    ik = IK()
    try:
        ok = grab_one(arm, A, xoff, yoff, ik, args.color)
    finally:
        ik.close()
    return 0 if ok else 1


def cmd_auto(args):
    """按 红->黄->蓝->绿 顺序连续抓取放置, 全程零交互, 一次到位."""
    order = ["red", "yellow", "blue", "green"]
    A, xoff, yoff = load_affine()
    arm = make_arm()
    ik = IK()
    placed = []
    try:
        for color in order:
            print("==== %s ====" % color)
            ok = grab_one(arm, A, xoff, yoff, ik, color)
            if ok:
                placed.append(color)
            else:
                print("跳过 %s (未检测到)" % color)
            time.sleep(0.4)
    finally:
        ik.close()
    # 回到拍照位收尾
    w6(arm, [91, 135, 0, 0, 90, RELEASE], 1500)
    print("ALL DONE placed=%s" % placed)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("detect", "grab"):
        p = sub.add_parser(name)
        p.add_argument("--color", default="any",
                       choices=["any", *COLORS.keys()])
    sub.add_parser("auto")      # 红黄蓝绿连抓连放, 零交互
    args = ap.parse_args()
    try:
        if args.cmd == "detect":
            return cmd_detect(args)
        if args.cmd == "auto":
            return cmd_auto(args)
        return cmd_grab(args)
    except Exception as e:
        print("ERROR", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
