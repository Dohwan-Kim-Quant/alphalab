from alphalab.operators import rank, ts_returns, ts_decay_linear, ts_std_dev

DAY = 24  # 1시간봉 기준 하루

def rev_smooth(d):
    """반전 신호를 하루에 걸쳐 부드럽게: 회전율 감소"""
    return ts_decay_linear(rank(-ts_returns(d.close, DAY)), DAY)

def rev_volscaled(d):
    """변동성으로 나눠서, 원래 많이 움직이는 코인의 영향 줄이기"""
    vol = ts_std_dev(d.returns, 7 * DAY)
    return rank(-ts_returns(d.close, DAY) / vol)

def rev_no_extremes(d):
    """상하위 극단 10%는 제외: 펌핑 코인 숏 회피"""
    r = rank(-ts_returns(d.close, DAY))
    return r.where((r > 0.1) & (r < 0.9))