# Dense packed R1：11/12 的第一階段

狀態：已加入程式，尚未在 Nano5 編譯或量測。a284817 的 sparse R1
曾通過 94 個小圖比對與 public 08/10；這不是新 dense wrapper 的驗證結果。

## 修改與原理

1. `step_r1_dense<Moore>` 讓每個 block 負責一塊 64×32 輸出 tile，
   共 256 threads。呼叫既有 `update_r1_packed<Moore, false>`：每個 thread
   處理兩個 32-bit words，合計八個細胞。四個 byte 各自累加 R1 鄰居數，
   最大為 9，沒有跨 byte 進位；仍保留每格一 byte 的完整 C2..16 狀態。
2. 既有 helper 先合作載入 state/alive 到 shared memory（4896 bytes/block），
   用 `__syncthreads()` 發布資料，再讀取鄰居。wrapper 不在邊界提前 return，
   否則可能破壞 block 內同步。有效輸出範圍由 helper 處理。
3. `Track=false` 在編譯時移除活動 reduction。dense 每塊都更新，無須 queue
   或 atomic。每代仍使用不同 global src/dst，kernel 同 stream 保證前後代順序。
4. dense grid 的 tile 尺寸一併從 naive 的 32×8 改為 64×32，確保覆蓋全圖且
   每個 block 的輸出互不重疊；shared memory 大小沿用 packed helper 的常數。
5. `run()` 在 dense 分支優先選擇 R1 packed，並輸出 `profile_kernel=dense_r1`。
   `PP_R1=direct` 可回到舊 dense naive 路徑；`PP_KERNEL=naive` 也保留 baseline
   語意，除非同時明確要求 `PP_R1=packed`。`PP_KERNEL=tiled` 仍強制 tiled。

11 的初始隨機全圖通常使 sparse 檢查拒絕稀疏路徑；12 的 128 MiB 大小
低於目前 256 MiB 的自動檢查門檻。因此兩者預期都走新 dense wrapper。
這不是 case ID 判斷：所有符合 R1 條件的 dense 輸入（包括 01-03）都會使用它。

## 解決與未解決的問題

- 解決：dense R1 原本沒有使用現成 packed helper，仍逐格計算鄰居。
- 解決：dense 使用固定分工，不必付出活動追蹤與工作清單的成本。
- 保留：R2-4 direct、R>=5 prefix sums、既有 sparse 排程及 fallback。
- 尚未解決：每代仍全圖讀寫；沒有 temporal blocking。
- 尚未解決：256 MiB / 128 代 / 25% 入場 / 50% fallback 是效能啟發式，
  不保證所有 hidden 的最佳選擇，也沒有 dense→sparse 的轉換。
- 均勻散布的低密度細胞可能讓所有 tiles 活動；dense packed 改善這種情況的
  全圖更新成本，但不會消除 tile 粒度過粗造成的額外工作。

## 最省額度的第一輪

在 login node 編譯，執行測試交給 Slurm：

```bash
make hw3-1 tests/test_hw3-1 &&
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 2 bash tests/run_dense_r1.sh profile
```

腳本會清除 PP_KERNEL / PP_ACTIVE / PP_R1 的外部設定，依序執行：

1. 預設派送的 90 個小圖逐格 CPU reference 比對：原 64 個 R1 邊界與規則
   組合，加 20 個 C2/C4、dead/torus、空圖/集中/分散/隨機/消退組合，
   加 6 個小圖 T=0/403/1190 測試。所有圖都低於 sparse 自動檢查門檻。
2. 強制 sparse 的既有 30 個 regression cases，含 R5、增長轉 dense 與衰退。
3. 只跑 public 11/12，使用官方 checker，且開啟分段量測。

每個 gate 最多 25 秒；每個 public process 的 wall timeout 是 15 秒。
腳本在 Slurm 外會直接拒絕執行。小圖 correctness 不是效能測試。

觀察 `profile_kernel=dense_r1`、`Compute time`（限時 0.8 秒）、`kernels_ms`。
11 可能額外顯示 `sparse_selected=0`，代表檢查後選擇 dense，屬於預期。
12 未進行 sparse 檢查，因此通常不會有 sparse 統計行。

結果存於 `tmp/dense-r1-<job>-<timestamp>-<pid>/default/results.tsv`。
程式結束時會印完整路徑。`TLE` 是 checker 已通過但時間超限；腳本允許這種
結果以便診斷，WA/RE/timeout 則中止。最後的 `Outputs verified correct`
不代表已通過時間限制，仍須看 AC/TLE。

先跑 profile 就好，不需要現在重跑全套 judge。

## 只有需要精確比較時才跑 A/B

```bash
srun -A ACD115218 -p dev -N1 -n1 -c4 \
  --gpus-per-node=1 -t 5 bash tests/run_dense_r1.sh ab
```

在同一 allocation 做 direct / packed / packed / direct；其他派送條件保持
自動，profiling 關閉，每輪都驗證輸出。`check` 模式則只做 gates 與新版
11/12 一輪（profiling 關閉）。歷史不同 job 的時間只能作參考，不當作受控加速比。

若新路徑仍超時，再依分段量測決定是否做 K=2 temporal blocking，
先不把排程門檻調整與跨代計算混在這次改動中。
