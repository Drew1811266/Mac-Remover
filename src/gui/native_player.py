"""
macOS 原生视频播放器封装（AVFoundation/AVKit）。

这个模块的目标是提供比浏览器内视频标签更稳定的预览能力：
1. 按角色（原始/处理后）管理独立播放器窗口。
2. 提供播放、暂停、跳转、状态查询等统一接口。
3. 强制在主线程执行 Cocoa UI 相关操作，避免线程问题。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any

from Foundation import NSURL
from CoreMedia import CMTimeMakeWithSeconds, CMTimeGetSeconds, kCMTimeZero
from AVFoundation import AVPlayer, AVPlayerTimeControlStatusPlaying, AVURLAsset, AVMediaTypeVideo
from AVKit import AVPlayerView
from AppKit import (
    NSApplication,
    NSWindow,
    NSBackingStoreBuffered,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskMiniaturizable,
    NSViewWidthSizable,
    NSViewHeightSizable,
    NSMakeRect,
)
from .main_thread_dispatch import run_on_main_sync


@dataclass
class PlayerSession:
    """保存一个播放器窗口会话的关键对象和元信息。"""

    role: str
    path: str
    title: str
    window: Any
    player_view: Any
    player: Any


class NativePlayerManager:
    """
    每个角色（`source` / `processed`）最多维护一个原生播放窗口。
    """

    def __init__(self):
        """初始化会话表和线程锁。"""
        self._sessions: Dict[str, PlayerSession] = {}
        self._lock = threading.RLock()

    def _run_on_main_sync(self, fn, *args, **kwargs):
        """
        同步地把函数派发到主线程执行。

        为什么需要它：
        - Cocoa/AVKit 的 UI 操作必须在主线程运行。
        - 这里用事件对象等待结果，并把异常原样抛回调用侧。
        """
        return run_on_main_sync(fn, *args, timeout_sec=5.0, **kwargs)

    def open(self, role: str, path: str, title: Optional[str] = None, autoplay: bool = False) -> Dict[str, Any]:
        """打开或复用某个角色的播放器窗口。"""
        if not role:
            role = "source"
        if not path:
            return {"success": False, "error": "Missing path"}
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        return self._run_on_main_sync(self._open_main, role, path, title, autoplay)

    def _open_main(self, role: str, path: str, title: Optional[str], autoplay: bool) -> Dict[str, Any]:
        """
        真正的主线程打开逻辑。

        - 首次打开：创建窗口 + AVPlayerView。
        - 再次打开：复用窗口，仅替换媒体源。
        """
        app = NSApplication.sharedApplication()
        app.activateIgnoringOtherApps_(True)

        role_title = title or ("原始视频预览" if role == "source" else "处理后视频预览")
        media_url = NSURL.fileURLWithPath_(path)
        player = AVPlayer.playerWithURL_(media_url)
        if player is None:
            return {"success": False, "error": "Failed to create AVPlayer"}

        session = self._sessions.get(role)
        if session is None:
            style = (
                NSWindowStyleMaskTitled
                | NSWindowStyleMaskClosable
                | NSWindowStyleMaskResizable
                | NSWindowStyleMaskMiniaturizable
            )
            window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(120, 120, 960, 560),
                style,
                NSBackingStoreBuffered,
                False,
            )
            window.setReleasedWhenClosed_(False)
            window.setTitle_(role_title)

            content = window.contentView()
            player_view = AVPlayerView.alloc().initWithFrame_(content.bounds())
            player_view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            player_view.setPlayer_(player)
            content.addSubview_(player_view)

            window.makeKeyAndOrderFront_(None)
            session = PlayerSession(
                role=role,
                path=path,
                title=role_title,
                window=window,
                player_view=player_view,
                player=player,
            )
            self._sessions[role] = session
        else:
            session.player_view.setPlayer_(player)
            session.player = player
            session.path = path
            session.title = role_title
            session.window.setTitle_(role_title)
            session.window.makeKeyAndOrderFront_(None)

        if autoplay:
            player.play()
        else:
            player.pause()

        duration = self._duration_seconds(session.player)
        fps = self._nominal_fps(path)
        return {
            "success": True,
            "role": role,
            "path": path,
            "duration": duration,
            "fps": fps,
        }

    def play(self, role: str) -> Dict[str, Any]:
        """开始播放指定角色的视频。"""
        return self._run_on_main_sync(self._play_main, role)

    def _play_main(self, role: str) -> Dict[str, Any]:
        """主线程播放实现。"""
        session = self._sessions.get(role)
        if session is None:
            return {"success": False, "error": f"No native player for role: {role}"}
        session.player.play()
        session.window.makeKeyAndOrderFront_(None)
        return {"success": True}

    def pause(self, role: str) -> Dict[str, Any]:
        """暂停指定角色的视频。"""
        return self._run_on_main_sync(self._pause_main, role)

    def _pause_main(self, role: str) -> Dict[str, Any]:
        """主线程暂停实现。"""
        session = self._sessions.get(role)
        if session is None:
            return {"success": False, "error": f"No native player for role: {role}"}
        session.player.pause()
        return {"success": True}

    def seek(self, role: str, seconds: float) -> Dict[str, Any]:
        """跳转到指定秒数（小于 0 会被钳制到 0）。"""
        return self._run_on_main_sync(self._seek_main, role, float(seconds))

    def _seek_main(self, role: str, seconds: float) -> Dict[str, Any]:
        """主线程跳转实现，使用零容差获得更精确定位。"""
        session = self._sessions.get(role)
        if session is None:
            return {"success": False, "error": f"No native player for role: {role}"}
        t = CMTimeMakeWithSeconds(max(0.0, seconds), 600)
        session.player.seekToTime_toleranceBefore_toleranceAfter_(t, kCMTimeZero, kCMTimeZero)
        return {"success": True}

    def state(self, role: str) -> Dict[str, Any]:
        """查询当前播放状态、位置和时长。"""
        return self._run_on_main_sync(self._state_main, role)

    def _state_main(self, role: str) -> Dict[str, Any]:
        """主线程状态查询实现。"""
        session = self._sessions.get(role)
        if session is None:
            return {"success": False, "error": f"No native player for role: {role}"}
        player = session.player
        position = CMTimeGetSeconds(player.currentTime())
        duration = self._duration_seconds(player)
        is_playing = bool(player.timeControlStatus() == AVPlayerTimeControlStatusPlaying)
        return {
            "success": True,
            "position": 0.0 if position is None else max(0.0, float(position)),
            "duration": max(0.0, float(duration)),
            "is_playing": is_playing,
        }

    def close(self, role: str) -> Dict[str, Any]:
        """关闭指定角色窗口；不存在时也返回成功。"""
        return self._run_on_main_sync(self._close_main, role)

    def _close_main(self, role: str) -> Dict[str, Any]:
        """主线程关闭实现，包含播放器暂停和窗口销毁。"""
        session = self._sessions.pop(role, None)
        if session is None:
            return {"success": True}
        try:
            session.player.pause()
        except Exception:
            pass
        try:
            session.window.orderOut_(None)
            session.window.close()
        except Exception:
            pass
        return {"success": True}

    def close_all(self) -> Dict[str, Any]:
        """关闭所有原生播放器窗口。"""
        return self._run_on_main_sync(self._close_all_main)

    def _close_all_main(self) -> Dict[str, Any]:
        """主线程批量关闭实现。"""
        roles = list(self._sessions.keys())
        for role in roles:
            self._close_main(role)
        return {"success": True}

    @staticmethod
    def _duration_seconds(player) -> float:
        """
        安全读取视频时长（秒）。

        这里对 `None`、NaN、负值做了兜底处理，避免污染前端状态。
        """
        try:
            item = player.currentItem()
            if item is None:
                return 0.0
            d = CMTimeGetSeconds(item.duration())
            if d is None:
                return 0.0
            if d != d:  # NaN check
                return 0.0
            if d < 0:
                return 0.0
            return float(d)
        except Exception:
            return 0.0

    @staticmethod
    def _nominal_fps(path: str) -> float:
        """
        从资源轨道读取名义帧率。

        读取失败时返回 0，调用方据此走“未知帧率”分支。
        """
        try:
            url = NSURL.fileURLWithPath_(path)
            asset = AVURLAsset.URLAssetWithURL_options_(url, None)
            tracks = asset.tracksWithMediaType_(AVMediaTypeVideo)
            if tracks and len(tracks) > 0:
                fps = float(tracks[0].nominalFrameRate())
                if fps > 0:
                    return fps
        except Exception:
            pass
        return 0.0
