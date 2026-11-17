# B1：共通rollout APIとaction chunk scheduler

更新日：2026-11-17。対象実装は `8354af2`（このリポジトリのshiba）。
既存のインストール環境で使う追加機能の説明です。環境構築は [インストール手順](install.md) を参照してください。

## 通常の運用

従来のrolloutコマンドをそのまま使います。schedulerは既定OFFで、
`--skip` ごとに推論する経路を維持します。ACT・GR00T・OpenPiの現在の
ALOHA運用では、`--use_action_chunk_scheduler` を追加しません。

B1はモデルや機体設定を提供しません。ALOHA側の共通変換は
[bridgeのC1マニュアル](https://github.com/TomohiroMOTODA/robomanip_aloha_bridge/blob/shiba/doc/shared_alignment_manual.md)
を参照してください。モデルごとのcheckpoint・入力順序・周期は別途一致させます。

## schedulerを試す場合

対応するpolicyの既存rolloutコマンドへ、次のオプションを追加します。
これは引数の一覧であり、単独で実行するコマンドではありません。

```text
--use_action_chunk_scheduler
--action_chunk_refill_threshold 0
--action_chunk_blend_steps 0
--action_chunk_max_queue_size 0
--action_chunk_underrun_policy hold
```

| 引数 | 既定値と動作 |
|---|---|
| use_action_chunk_scheduler | 未指定でOFF。指定時はキューから各制御tickのactionを取得 |
| action_chunk_refill_threshold | 0。残数が閾値以下なら同期推論して追加 |
| action_chunk_blend_steps | 0。正数なら境界直後の指定数を前のactionから線形blend |
| action_chunk_max_queue_size | 0で上限なし。正数なら上限超過分を末尾から削除 |
| action_chunk_underrun_policy | hold。空キューでは直前値を保持。初回値がない場合、またはerror指定なら例外 |

推論は同期実行です。非同期推論・通信遅延補償・学習周期から制御周期への補間は行いません。
既定のchunk取得は `policy_action` を使います。policyが単一actionを保持するか、
chunk全体を保持するかを確認し、必要なら `get_policy_action_chunk_for_scheduler()` を実装してください。
RTCやpolicy独自のchunk消費経路との併用は、各policy側で別途検証が必要です。

終了時に表示されるscheduler statisticsでpush/pop、underrun、queue depth、
境界変化量、推論時間を確認します。統計だけでは実機の動作改善を判断できません。
元に戻すにはプロセスを終了し、scheduler引数を外して起動します。
実機rolloutの起動・resetには移動が伴う場合があるため、既存の機体手順に従ってください。

## policy実装者向け

- `reset_plot()`：reset時に動的呼出しします。Qt等の独自描画ではoverrideしてください。
  既定実装は従来のMatplotlib描画です。`--no_plot` 時は呼びません。
- `infer_policy_with_timing()`：inference modeで推論し、時間を記録します。
- `get_policy_action_chunk_for_scheduler()`：キューへ渡すaction配列を返します。
- `record_scheduler_policy_action()`：実際に消費したactionを履歴へ反映します。
- `MotionManagerClass` / `DataManagerClass`：既存の拡張点を維持します。
- resetでは前episodeのキュー・直前action・統計をクリアします。

## 実機を使わない動作確認

このリポジトリのルートで、依存関係を導入したPythonを使います。
次の例はキュー内の配列だけを操作し、ロボットへ接続しません。

```bash
python - <<'PY'
import numpy as np
from robo_manip_baselines.common.manager.ActionChunkScheduler import ActionChunkScheduler
queue = ActionChunkScheduler(action_dim=2)
queue.push([[1., 2.], [3., 4.]])
np.testing.assert_array_equal(queue.next_action(), [1., 2.])
np.testing.assert_array_equal(queue.next_action(), [3., 4.])
np.testing.assert_array_equal(queue.next_action(), [3., 4.])  # hold
assert queue.get_statistics()['underrun_count'] == 1
queue.reset()
assert queue.queue_depth == 0 and queue.last_action is None
print('B1 queue check: OK')
PY
```

公開前の検証では、master＋B1だけの分離コピーで共通API互換性3件と一時キュー動作確認4件が成功しました。
共通APIテストはbridge側のレビュー対象資料を使用したもので、B1コミットには同梱していません。
このマニュアルの追加ではロボット操作・ビルドを行っていません。
