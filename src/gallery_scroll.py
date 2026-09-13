"""共用:讓「所有」gr.Gallery 元件開啟預覽(lightbox)後,底部的縮圖列
(.thumbnails)可以捲軸拖曳 + 滾輪橫向捲動 + 按住拖曳。

不綁定特定 elem_id、對整個 App 生效 —— 因為 Gradio 的預覽縮圖列
是同一套原生元件、同一套 class 名稱,不管是哪個分頁的哪個 Gallery
(抽幀結果、批次標註結果、去重結果、蒸餾預覽...),結構都一樣,
沒必要每個分頁各刻一份。

CSS 可以透過 gr.HTML() 注入(style 標籤在 innerHTML 裡照樣生效)。
JS 不行——瀏覽器規範:透過 innerHTML 插入的 <script> 天生不會執行,
一定要透過 demo.launch(head=...) 寫進真正的 <head> 才會跑,
所以 GALLERY_SCROLL_HEAD_JS 要由 app.py 集中傳給 launch(head=...)。
"""

GALLERY_SCROLL_CSS = """
<style>
/* Gradio 原生用 .scroll-hide 把預覽縮圖列的捲軸藏起來,這裡蓋掉讓它可見 */
.thumbnails {
    cursor: grab;
    /* Firefox: 原生只有 auto/thin/none 三檔,沒有 px 可調,用 thin 逼近需求 */
    scrollbar-width: thin !important;
    scrollbar-color: #888 #2b2b2b !important;
}
.thumbnails.ff-grabbing {
    cursor: grabbing;
}
.thumbnails::-webkit-scrollbar {
    display: block !important;
    height: 12px !important;
}
.thumbnails::-webkit-scrollbar-track {
    background: #2b2b2b;
}
.thumbnails::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 6px;
}
.thumbnails::-webkit-scrollbar-thumb:hover {
    background: #aaa;
}
</style>
"""

GALLERY_SCROLL_HEAD_JS = """
<script>
(function() {
    // 事件委派掛在 document 上,不管是哪個分頁、哪個 Gallery,
    // 也不管 Gradio 重新渲染幾次把 .thumbnails 節點整個換掉,
    // 都不需要重新綁定——e.target.closest() 在事件發生的當下
    // 即時去抓目前畫面上的 .thumbnails,永遠抓得到最新的節點。
    function findThumbs(target) {
        return (target && target.closest) ? target.closest('.thumbnails') : null;
    }

    document.addEventListener('wheel', function(e) {
        var thumbs = findThumbs(e.target);
        if (!thumbs) return;
        if (e.deltaY === 0) return;
        thumbs.scrollLeft += e.deltaY;
        e.preventDefault();
    }, { passive: false });

    var dragState = null;
    var justDragged = false;

    document.addEventListener('mousedown', function(e) {
        var thumbs = findThumbs(e.target);
        if (!thumbs) return;
        dragState = { thumbs: thumbs, startX: e.pageX, startScroll: thumbs.scrollLeft, moved: false };
        thumbs.classList.add('ff-grabbing');
    });

    document.addEventListener('mousemove', function(e) {
        if (!dragState) return;
        var dx = e.pageX - dragState.startX;
        if (Math.abs(dx) > 3) dragState.moved = true;
        dragState.thumbs.scrollLeft = dragState.startScroll - dx;
    });

    document.addEventListener('mouseup', function() {
        if (dragState) {
            dragState.thumbs.classList.remove('ff-grabbing');
            justDragged = dragState.moved;
            dragState = null;
        }
    });

    // 真的有拖曳位移時,吃掉放開瞬間緊接著的 click,
    // 避免放開時誤觸縮圖切換(既有的點圖預覽/選取行為完全不受影響)。
    document.addEventListener('click', function(e) {
        var thumbs = findThumbs(e.target);
        if (!thumbs) return;
        if (justDragged) {
            justDragged = false;
            e.preventDefault();
            e.stopPropagation();
        }
    }, true);
})();
</script>
"""
