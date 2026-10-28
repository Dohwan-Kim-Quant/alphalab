# alphalab

> **AI 사용 고지**
> 초기 구조(데이터 파이프라인, BRAIN 스타일 연산자, 백테스터, 테스트)는 Claude(AI)의 도움을 받아 만들었습니다(첫 커밋).
> 이후 실험 설계, 결과 해석, 코드 수정은 제가 직접 진행했으며, 커밋 이력으로 구분됩니다.

바이낸스 USDⓈ-M 무기한 선물 데이터 파이프라인과, WorldQuant BRAIN 스타일의 횡단면(cross-sectional) 알파 리서치 도구입니다.

**연구 질문:** BRAIN(주식)에서 만든 알파 아이디어 중 어떤 것이 크립토 선물로 이식되고, 어떤 것은 죽는가?

```
data.binance.vision ──► historical.py ─┐
  (월/일 단위 아카이브)                 ├──► Parquet (심볼별 파일) ──► data.py ──► alphas.py ──► backtest.py ──► report
fstream/fapi.binance.com ► realtime.py ┘                              (정렬된 패널)  (BRAIN 연산자)   (비용·펀딩 반영)
```

---

## 설치

Python 3.10 이상.

```bash
cd alphalab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 사용법

### 0. 먼저 데모로 동작 확인 (네트워크 불필요)

```bash
alphalab demo
```

일부러 "1일 반전" 신호를 심어둔 가상 시장에서 `reversal_1d` 알파를 돌립니다. 신호를 찾아내면(IC > 0, 그로스 샤프 양수) 백테스터가 정상입니다. 수수료를 빼면 샤프가 크게 줄어드는 것도 함께 보입니다. 회전율이 높은 알파가 실전에서 왜 어려운지가 이 숫자에 그대로 드러납니다.

### 1. 과거 데이터 수집

```bash
# 2021년부터 모든 USDT 무기한 선물 (상장폐지 종목 포함), 1시간봉 + 펀딩비
alphalab download --interval 1h --start 2021-01-01

# 일부 종목만
alphalab download --interval 1h --start 2023-01-01 --symbols BTCUSDT ETHUSDT SOLUSDT
```

다시 실행하면 새로 생긴 파일만 받습니다 (`data/_manifest/` 참고).

### 2. 실시간 수집

```bash
alphalab realtime --interval 1h --top 100     # 24시간 거래대금 상위 100개
```

Ctrl+C로 종료하면 버퍼에 남은 봉까지 저장한 뒤 멈춥니다. **과거 수집과 실시간 수집을 같은 interval로 동시에 돌리지 마세요** (같은 파일에 두 프로세스가 쓰면 마지막 쓰기만 남습니다).

### 3. 백테스트

```bash
alphalab list-alphas
alphalab backtest --alpha reversal_1d --top-n 50
alphalab backtest --alpha momentum_30d_skip_1d --neutralize beta --start 2022-01-01
alphalab backtest --alpha my_alphas.py:my_alpha        # 직접 짠 알파 파일
```

결과는 `results/<알파이름>/`에 `pnl.csv`, `yearly.csv`, `summary.json`, `equity.png`로 저장됩니다.

### 직접 알파 작성하기

```python
# my_alphas.py
from alphalab.operators import rank, ts_corr, ts_delta, ts_mean

def my_alpha(d):
    # BRAIN:  rank(-ts_delta(close, 3)) * rank(volume / ts_mean(volume, 20))
    day = 24  # 1시간봉 기준 하루 = 24봉
    return rank(-ts_delta(d.close, 3 * day)) * rank(d.volume / ts_mean(d.volume, 20 * day))
```

`d`는 `MarketData`이며 `open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, returns, vwap, taker_buy_ratio, funding_rate`를 제공합니다. 모두 행=시간, 열=심볼인 DataFrame입니다.

**주의:** BRAIN의 윈도우는 "일" 단위지만 여기서는 "봉" 단위입니다. 1시간봉이면 `ts_mean(x, 20)`은 20시간입니다.

---

## 타이밍 규약 (면접에서 가장 많이 찔리는 부분)

행 `t`는 `t`에 열려서 `t + interval`에 닫히는 봉입니다.

| 항목 | 정의 |
|---|---|
| `alpha[t]` | 봉 `t` 종가까지의 데이터로 계산 |
| 목표 비중 `W[t]` | `alpha[t]`를 중립화·정규화한 값 |
| 포지션 `P[t]` | `W[t - delay]`, 봉 `t` 종가 직후 보유 |
| 봉 `t`의 손익 | `P[t-1] · r[t]`, `r[t] = close[t]/close[t-1] - 1` |
| 회전율 `t` | `Σ|P[t] - P[t-1]|`, 봉 `t` 종가에 체결, `fee_bps` 부과 |
| 펀딩 `t` | `-P[t-1] · f[t]`, 봉 `t` 안에서 정산된 펀딩 (양수면 롱이 지불) |

- `delay=0`은 신호를 계산한 그 종가에 바로 체결한다는 가정이라 낙관적입니다. 기본값은 `delay=1`.
- 08:00 펀딩 정산은 1시간봉 기준 07:00 봉에 귀속됩니다. 08:00까지 들고 있던 사람이 냅니다.
- 손익은 그로스 1 기준입니다 (시장중립이면 롱 0.5 + 숏 0.5). BRAIN처럼 수익률을 복리가 아닌 단순 합산합니다.

이 규약은 `tests/test_backtest.py`에서 손으로 계산한 값과 비교해 검증합니다. 특히 `test_future_peeking_alpha_only_works_without_delay`는 "다음 봉 수익률을 아는" 반칙 알파가 `delay=0`에서만 돈을 벌고 `delay=1`에서는 못 버는지 확인합니다. 타이밍이 한 칸이라도 어긋나면 실패합니다.

## 설계 결정과 이유

| 결정 | 이유 |
|---|---|
| 상장폐지 종목까지 수집 | 지금 살아있는 코인만으로 백테스트하면 죽은 코인이 빠져서 성과가 부풀려집니다 (생존 편향). 아카이브 목록에서 심볼을 찾으면 폐지 종목도 포함됩니다. |
| 월간 파일 + 이번 달은 일간 파일 | 요청 수를 줄이면서 최신 데이터까지 채웁니다. 월간 파일이 나오면 같은 타임스탬프를 덮어써서 중복이 없습니다. |
| SHA-256 체크섬 검증 | 깨진 다운로드가 조용히 데이터에 섞이는 걸 막습니다. 검증 실패 파일은 저장하지 않고 다음 실행 때 다시 받습니다. |
| 매니페스트는 저장 성공 후 기록 | 저장 중 중단되면 해당 파일을 다시 받습니다. 순서가 반대면 데이터 구멍이 생깁니다. |
| 원자적 쓰기 (임시 파일 → `os.replace`) | 프로세스가 중간에 죽어도 반쯤 쓰인 파일이 남지 않습니다. |
| 웹소켓은 닫힌 봉만 저장 | 형성 중인 봉을 저장하면 백테스트에 존재하지 않았던 가격이 섞입니다. |
| 재연결 시 REST 백필 | 끊긴 동안 놓친 봉을 메웁니다. 구독을 먼저 열고 백필해서 그 사이 틈이 생기지 않게 했고, 겹치는 봉은 저장 시 중복 제거됩니다. |
| 24시간 전에 연결 교체 | 바이낸스는 연결을 24시간 후 끊습니다. 미리 교체하고 백필로 이음새를 메웁니다. |
| 디스크 쓰기는 별도 스레드 | `asyncio.to_thread`로 처리해 파일 쓰기가 이벤트 루프(메시지 수신)를 막지 않게 했습니다. |
| 패널을 규칙적인 시간 격자로 정렬 | 데이터에 빈 봉이 있어도 "d봉 전"이 모든 심볼에서 같은 시각을 가리킵니다. |
| 시계열 연산자는 윈도우가 꽉 차야 값 계산 | 짧은 윈도우로 조용히 계산되는 걸 막습니다. 모든 연산자가 미래를 보지 않는지 테스트로 확인합니다. |
| 유니버스는 과거 거래대금 기준 상위 N, 최소 상장 기간 | 매 시점에 그 시점까지의 정보만 씁니다. 상장 직후의 극단적인 움직임도 제외합니다. |
| 베타 중립화 옵션 | 크립토는 대부분 BTC를 따라 움직여서, 금액 중립만으로는 고베타 코인 롱/저베타 코인 숏이 숨은 시장 베팅이 됩니다. `--neutralize beta`는 Σw = 0과 Σw·β = 0을 동시에 맞춥니다. |
| 종목당 비중 상한 (water-filling) | 상한을 넘는 종목은 상한에 고정하고 나머지를 비례 배분해서 롱/숏 그로스를 유지합니다. 한쪽 종목 수가 부족하면 반대쪽도 줄여 중립을 지킵니다. |
| 네트워크는 함수 주입 | 다운로드·수집 로직을 네트워크 없이 테스트할 수 있습니다. |

## BRAIN → 크립토 이식 시 체크포인트

- **유니버스 크기**: BRAIN은 수천 종목, 바이낸스 선물에서 유동성 있는 건 수십~백여 개입니다. 같은 IC여도 분산 효과가 작습니다.
- **시장 베타**: 크립토는 종목 간 상관이 매우 높습니다. `market`과 `beta` 중립화를 둘 다 돌려서 결과가 얼마나 달라지는지 보세요.
- **비용**: 시간봉 리밸런싱은 회전율이 매우 높습니다. 결과의 `gross | fees | funding` 분해에서 수수료가 그로스를 얼마나 먹는지 먼저 보세요.
- **펀딩비**: 주식에 없는 비용이자 신호입니다 (`funding_carry`).
- **24/7 시장**: 연율화는 365일 × 24시간 기준입니다.
- **없는 데이터**: 재무제표·애널리스트 데이터 기반 BRAIN 알파는 이식할 수 없습니다.

## 알려진 한계

- 체결은 종가 기준, 슬리피지는 `fee_bps`에 합쳐서 근사합니다. 오더북 깊이를 고려한 시장 충격 모델은 없습니다.
- 상장폐지 직전 마지막 가격 이후의 수익률은 0으로 처리합니다. 실제로는 정산가와 차이가 날 수 있습니다.
- 이번 달 펀딩비는 실시간 수집기를 돌려야 채워집니다 (아카이브는 월 단위로만 공개).
- 바이낸스는 웹소켓 엔드포인트 구성을 바꿔왔습니다. 연결이 안 되면 공식 문서에서 현재 USDⓈ-M 주소를 확인하고 `--ws-base`로 지정하세요.
- 아카이브 목록은 `s3-ap-northeast-1.amazonaws.com/data.binance.vision`을 사용합니다. 바이낸스가 이를 바꾸면 `historical.LIST_URL`을 수정해야 합니다.

## 테스트

```bash
pytest -q
```

네트워크 없이 돌아갑니다. 가짜 아카이브, 가짜 REST/웹소켓, 가상 시장을 사용합니다.

## 첫 실행 체크리스트

1. `alphalab demo`가 양수 IC를 보이는지
2. `alphalab download --symbols BTCUSDT ETHUSDT --start 2024-01-01`로 소량 먼저 받아보기
3. `data/klines/1h/BTCUSDT.parquet`을 pandas로 열어 시간 간격, 결측, 가격 범위 확인
4. 전체 다운로드 후 `alphalab backtest --alpha reversal_1d`
5. 실시간 수집은 과거 데이터를 받은 뒤 시작 (처음 보는 심볼은 최근 7일만 백필)

## 면접 대비: 설명할 수 있어야 하는 질문들

- 생존 편향을 어떻게 처리했나? 왜 그게 중요한가?
- `delay=0`과 `delay=1`의 차이는? 내 알파는 둘 사이에서 얼마나 차이 나나?
- 그로스 샤프와 넷 샤프 차이가 크면 무엇을 의미하나? 회전율을 줄이려면 (`ts_decay_linear`, 리밸런싱 주기 늘리기 등)?
- 웹소켓이 끊겼다 다시 붙으면 데이터 구멍은 어떻게 막나?
- 왜 시장 중립이 아니라 베타 중립이 필요할 수 있나?
- 여러 알파를 시험했다면 다중검정 문제는 어떻게 고려했나? (기간을 나눠 검증하기, 시험한 알파 수를 기록하기)
- 이 결과가 실제 운용에서 재현되지 않을 이유를 세 가지 들어보라.

## 구조

```
src/alphalab/
  historical.py   아카이브 목록·다운로드·체크섬·파싱·증분 수집
  realtime.py     웹소켓 수집기, REST 백필, 펀딩비 폴링
  store.py        심볼별 Parquet 저장, 원자적 upsert, 매니페스트
  data.py         정렬된 패널(MarketData), 펀딩 정산의 봉 귀속
  operators.py    BRAIN 스타일 연산자 (rank, ts_rank, ts_corr, decay, trade_when ...)
  universe.py     시점별 유동성 유니버스
  backtest.py     비중 생성(중립화·상한), 손익·비용·펀딩, IC, 요약 지표
  alphas.py       예시 알파와 레지스트리
  report.py       요약 출력, 수익 곡선 그림
  synthetic.py    신호를 심어둔 가상 시장 (테스트·데모용)
  cli.py          명령줄 진입점
tests/            네트워크 없이 돌아가는 단위·통합 테스트
```
