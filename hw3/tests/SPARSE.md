# Sparse 實作導讀

本版為 08～10 這類大圖、多代、局部活動問題加入工作清單。演算法根據
規則與實際非零狀態選擇路徑，不依 case 編號或種子。原 sparse/direct 版
已通過 08～10 的答案檢查，其中 09 過時限；新 packed R1 版尚待 GPU 驗證。

## 1. 分開「計數」與「排程」

`hw3-1.cu` 的 `update_tiled<Moore, Track>` 從原本 `step_tiled` 抽出。
`step_tiled` 保留原本的 dense grid 啟動方式；`step_sparse` 則從工作清單
取得 tile 座標，再呼叫同一個計數函式。`Track=false` 時編譯器不產生
非零追蹤的兩次 block reduction。

原本小半徑 sparse 使用 `update_direct<Track>`，一個 block 的 256 threads
共同覆蓋 64×32 格，每 thread 最多更新 8 格。每格寫入位置互不重疊，
細胞資料不需要 atomic。大半徑仍使用原有 shared-memory prefix sums。
最新 R1 替換路徑見第 9 節；direct 路徑保留供比較。

## 2. 先建立正確的 tile 影響範圍

`tile_geometry.hpp::influence()` 將一塊 tile 的實際格子區間向兩側擴張 R，
再找出碰到的 tile 編號。Torus 先繞回格子座標，再換算 tile 編號。

例如寬度 321、tile 寬度 64，最後一塊只有一欄。R=5 從左邊跨界時，
可能越過最後一塊，繼續影響倒數第二塊；只取 tile 編號的左右鄰居會漏算。
菱形也使用矩形擴張作為保守上界，可能多算，但不漏算。

這個小函式以 `__host__ __device__` 同時供 CUDA 與 CPU 測試使用。
CPU 測試的 oracle 是逐一枚舉格子座標，不是重寫相同的區間演算法。

## 3. GPU 判斷初始工作量

`mark_initial_tiles` 每 block 掃一個 tile，以 `__syncthreads_or` 合併
「有沒有非零格」；`seed_worklist` 檢查影響範圍，產生第一次工作清單。

`state != 0` 包含 C3/C16 的衰退狀態；只有 `state == 1` 算活鄰居。
兩個條件用途不同，不能混用。CPU 只讀取一個工作數字來選擇排程，
不計算任何細胞的下一代。

`PP_ACTIVE=auto` 先要求至少 2^28 格與 128 代，才支付這次偵測成本。
初始工作清單不超過全部 tiles 的 25% 才進入 sparse。這些門檻尚未調校。
`PP_ACTIVE=0` 固定 dense；`PP_ACTIVE=1` 強制 sparse 起步，供小圖測試。

## 4. 固定數量 blocks，循環處理工作清單

`step_sparse` 使用 block-stride loop：

```cpp
for (int i = blockIdx.x; i < count; i += gridDim.x) {
    int tile = current.ids[i];
    // 整個 block 合作更新 tile
}
```

blocks 起始數量取 `min(tile_count, 4*SM_count)`，交給硬體安排駐留，
不是假設每個 SM 必定同時執行四個 blocks。每代不把 count 傳回 CPU。
同 block 的 threads 必須一致走完迭代；迭代尾端 barrier 確保上一塊的
shared memory 已讀完，下一塊才能覆寫它。

## 5. 下一代清單同時負責活動擴張與舊值清除

每個 thread 累積 input/output 非零旗標，兩次 `__syncthreads_or` 將旗標
合併成整個 block 的結果。代表 thread 執行：

```cpp
if (flags & 2) enqueue_influence(...); // 輸出非零：通知受影響區塊
if (flags & 1) enqueue_tile(...);      // 輸入非零：下次清除舊 buffer
```

令 A_t 為第 t 代非零 tiles，更新第 t+1 代的清單為
`W_t = dilation(A_t) ∪ A_(t-1)`。目的 buffer 還保存 t−1 代，第二項是
避免死亡細胞因未覆寫而復活的關鍵。初始化只清零一次目的 buffer，之後
透過這個不變量保持被跳過的目的區域為零。

## 6. Atomic 與 kernel 順序各有責任

`enqueue_tile` 用 `atomicExch(stamp + tile, epoch)` 選出唯一加入者，再以
`atomicAdd(count, 1)` 保留清單位置。每代 stamp 值不同，不必每代清除
整張 stamp 陣列。清單容量等於全部 tile 數，去重後不會超過容量。

Atomic 解決「多個 producer 搶同一個 tile／清單位置」，並不保證其他
blocks 已經把清單寫好。本版等本代 kernel 完成，下一代 kernel 才讀清單，
利用同 stream 的工作順序建立 producer/consumer 邊界。

`prepare_sparse<<<1,1>>>` 在 worker 前清零下一份 count。若放進 worker
的某一個 block，其他 blocks 可能已開始加入清單，造成競爭，因此分開。
每代兩個 kernels，但沒有每代 CPU/GPU 同步。

## 7. 避免 dense 資料持續付清單成本

`prepare_sparse` 發現工作超過全部 tiles 的 50%，就設定 device 上的 dense
旗標。後續 worker 直接遍歷全部 tiles，使用 `Track=false`，停止追蹤與
入列 atomic。即使 `PP_ACTIVE=1` 強制起步，仍保留此退路。

這是單向切換；第一版不再掃描以重新切回 sparse。25% 的進入門檻與 50%
的退出門檻留下餘裕，但都只是啟發式。是否有幫助要看實測。

## 8. 驗證與量測

`test_sparse_logic.cpp` 在 CPU 驗證幾何與清單不變量；它不能驗證 GPU
barriers／atomics。`test_hw3-1 --sparse` 才會執行真正 CUDA 路徑，逐格比對
原始 CPU reference，涵蓋 30 個跨 tile、跨 torus、衰退、消失、成長情境。

`run_sparse.sh profile` 先跑 74 次小圖 reference 比對，正確才跑 08～10，
使用與 judge 相同的預設選擇。完整命令與紀錄欄位見 [PERF.md](PERF.md)。

`sparse_mean_work / sparse_tiles_total` 是平均更新比例；不是活細胞密度。
`sparse_dense_steps` 可判斷後來是否轉為 dense。正式 Compute time 才是
判斷是否過時限的指標，不能只看工作比例。

目前仍完整 H2D／D2H，未加入 CUDA Graphs 或選擇性回傳。

## 9. R1：shared memory 與四個 byte 的並行計數

Job 402205 確認 08／10 的 sparse 有啟用、平均工作比例僅 1.76%／7.32%，
沒有切回 dense，但 kernel 區間仍佔 95～97%。這個區間包含排程，尚不能
斷言全是鄰居計數成本。新候選先降低區塊內計算成本，保留排程供比較。

`update_r1_packed` 將四個相鄰格子放入一個 uint32_t，各佔一個 byte。
先合作載入 tile 加 halo，再把 `state==1` 轉成每 byte 一個 0/1 指示值。
透過位移對齊左右鄰居、整數加法累加上下列，一次求出四格的鄰居數。
每 byte 的總和最多為 9，不會進位到旁邊的格子。這是 SWAR 的用法：
在一個 register 裡安排多個獨立的小整數運算；grid 本身仍是一格一 byte。

`r1_packed.hpp::in_range` 對每個 byte 加上高位哨兵，再相減比較上下界，
避免跨 byte 借位。`next` 保留 C2..C16 的出生、存活與衰退規則。這些純
運算函式由 CPU 與 CUDA 共用，已做 20,079,304 次 scalar 對照。

一個 tile 的 state/alive shared tables 合計 4896 bytes。輸出仍是每 thread
八格，但改成兩個 word。寬度為四的倍數時用對齊的 32-bit load/store；
其他寬度逐 byte 打包，尾端不合法的 bytes 不得寫出或納入非零旗標。

`PP_R1=direct` 可重現舊計數；預設在 sparse R1 路徑使用 packed，profile
標籤為 `sparse_r1`。R5 前綴和與 sparse 管理邏輯不變。新的 `run_r1.sh`
先跑 94 次小圖 reference 比對，再只量 08／10。執行命令見 PERF.md。

繳交時需附上 `tile_geometry.hpp` 與 `r1_packed.hpp`。
