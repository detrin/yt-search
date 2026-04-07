import time


class CentrifugalGovernor:
    """
    Circuit breaker modelled after Watt's centrifugal governor.

    Each failed video download pushes the balls higher (overspeed_surge).
    Once they swing too high the linkage cuts the steam — no more requests
    until the engine spins back down (spindown_seconds).  A successful
    download drops the balls back to rest (steady_state).
    """

    def __init__(self, max_swing_height: int, spindown_seconds: float):
        self._current_swing = 0
        self._steam_cut_off_until = 0.0
        self._max_swing_height = max_swing_height
        self._spindown_seconds = spindown_seconds

    @property
    def spindown_seconds(self) -> float:
        return self._spindown_seconds

    def choked_for(self) -> float:
        """Seconds until steam is restored; 0.0 if currently open."""
        return max(0.0, self._steam_cut_off_until - time.monotonic())

    def is_choked(self) -> bool:
        return self.choked_for() > 0.0

    def wait_if_choked(self) -> None:
        delay = self.choked_for()
        if delay > 0.0:
            print(f"  [governor] rate-limited — waiting {delay:.0f}s for spindown...")
            time.sleep(delay)

    def steady_state(self) -> None:
        """Successful download: gravity drops the balls back to rest."""
        self._current_swing = 0

    def overspeed_surge(self) -> None:
        """Failed download: centrifugal force pushes the balls higher."""
        self._current_swing += 1
        if self._current_swing >= self._max_swing_height:
            self._steam_cut_off_until = time.monotonic() + self._spindown_seconds
