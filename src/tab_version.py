"""版本管理:列出可用版本 tag、切換/回溯、相依套件同步。

沒有 .git 資料夾(zip 分發版本)時整個功能優雅隱藏——用 demo.load() 在頁面
載入後判斷,不是靠 import 時就檢查(那樣會讓整頁在非 git 環境直接報錯)。
"""
import gradio as gr

import git_utils
import version_switch as vs
from __version__ import __version__ as CURRENT_VERSION


def _status_html(is_rollback, label):
    if is_rollback:
        return (
            f'<div style="background:#5c1a1a;color:#ffb4b4;padding:10px 14px;'
            f'border-radius:8px;font-weight:600;">⚠️ 目前回溯到舊版本:{label}'
            f'(detached HEAD)——請用下面「回到最新版」按鈕離開這個狀態</div>'
        )
    return (
        f'<div style="background:#123d1f;color:#9ef5b0;padding:10px 14px;'
        f'border-radius:8px;font-weight:600;">✅ 目前在最新版:{label}</div>'
    )


def _current_status_html():
    branch = vs.get_current_branch()
    if branch == "main":
        return _status_html(False, f"main / v{CURRENT_VERSION}")
    tag = vs.get_current_tag()
    return _status_html(True, tag or "(未知位置)")


def _tag_name_from_label(label):
    return label.split(" ", 1)[0] if label else None


def _on_load():
    """demo.load() 用:回傳 (not_git_msg, main_panel, status_html, tag_dd) 4 個 update。"""
    if not git_utils.is_git_repo():
        return (
            gr.update(visible=True,
                     value="此功能需要 git 工作目錄,zip 分發版本不支援版本切換(這是正常狀況,不是錯誤)。"),
            gr.update(visible=False),
            gr.update(),
            gr.update(choices=[]),
        )

    tags = vs.list_tags()
    choices = [f"{t['name']} ({t['date']})" for t in tags]
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=_current_status_html()),
        gr.update(choices=choices),
    )


def _on_refresh():
    tags = vs.list_tags()
    choices = [f"{t['name']} ({t['date']})" for t in tags]
    return gr.update(value=_current_status_html()), gr.update(choices=choices)


def _on_select_tag(label):
    tag = _tag_name_from_label(label)
    if not tag:
        return ""
    return vs.changelog_summary(tag) or "(這個版本沒有對應的 CHANGELOG 段落)"


def _on_switch(label):
    tag = _tag_name_from_label(label)
    if not tag:
        return "請先選一個版本。", gr.update(), ""
    r = vs.switch_to_tag(tag)
    if r["status"] == "success":
        html = _status_html(True, tag)
        reminder = ("⚠️ 提醒:模型權重(models/*.pt)不在 repo 裡,這個版本需要的權重"
                   "請自行到對應的 GitHub Release 頁面確認/下載,不會自動幫你切換。")
        return r["message"], gr.update(value=html), reminder
    return r["message"], gr.update(), ""


def _on_back_to_latest():
    r = vs.switch_to_latest()
    if r["status"] == "success":
        return r["message"], gr.update(value=_status_html(False, f"main / v{CURRENT_VERSION}"))
    return r["message"], gr.update()


def _on_check_deps(label):
    tag = _tag_name_from_label(label) or "HEAD"
    r = vs.diff_requirements(tag)
    if "error" in r:
        return f"⚠️ {r['error']}", gr.update(visible=False)
    missing = r["missing"]
    if not missing:
        return "✅ 相依套件跟這個版本的 requirements.txt 一致,沒有缺少的套件。", gr.update(visible=False)
    return (f"⚠️ 這個版本的 requirements.txt 裡,目前 venv 缺少以下套件:\n"
            + "\n".join(f"  - {m}" for m in missing)), gr.update(visible=True)


def _on_sync_deps():
    r = vs.sync_requirements()
    return r["message"]


def render_ui():
    gr.Markdown("## 版本管理")
    gr.Markdown(
        "列出所有已發佈的版本,可以切換回舊版本檢查問題,或用「回到最新版」隨時回來。"
        "**切換版本後需要重新啟動程式**才會生效。"
    )

    not_git_msg = gr.Markdown(visible=False)

    with gr.Group(visible=False) as main_panel:
        status_html = gr.HTML()

        with gr.Row():
            tag_dd = gr.Dropdown(label="選擇版本", choices=[], interactive=True)
            refresh_btn = gr.Button("重新整理版本清單", scale=0)

        changelog_box = gr.Textbox(label="這個版本的 CHANGELOG 摘要", lines=6, interactive=False)

        with gr.Row():
            switch_btn = gr.Button("切換到這個版本", variant="primary")
            latest_btn = gr.Button("🏠 回到最新版", variant="stop")

        switch_status = gr.Textbox(label="切換結果", lines=3)
        model_reminder = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown("### 相依套件檢查")
        with gr.Row():
            check_deps_btn = gr.Button("比對這個版本的 requirements.txt")
            sync_deps_btn = gr.Button("同步相依套件(pip install -r requirements.txt)", visible=False)
        deps_status = gr.Textbox(label="套件比對結果", lines=6)

    refresh_btn.click(fn=_on_refresh, inputs=None, outputs=[status_html, tag_dd])
    tag_dd.change(fn=_on_select_tag, inputs=[tag_dd], outputs=[changelog_box])
    switch_btn.click(fn=_on_switch, inputs=[tag_dd],
                     outputs=[switch_status, status_html, model_reminder])
    latest_btn.click(fn=_on_back_to_latest, inputs=None, outputs=[switch_status, status_html])
    check_deps_btn.click(fn=_on_check_deps, inputs=[tag_dd],
                         outputs=[deps_status, sync_deps_btn])
    sync_deps_btn.click(fn=_on_sync_deps, inputs=None, outputs=[deps_status])

    return _on_load, [not_git_msg, main_panel, status_html, tag_dd]
