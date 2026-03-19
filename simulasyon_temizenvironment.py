"""
Beer Production System – SimPy Simulation
==========================================
Topology (left → right):
  Silos (M,N,O,P,R)
    → Malt Transfer Lines (Cimbria, Kunzel, Alapros)
    → Mills (Millstar, Powermill, Variomill)
    → Mashing (1-5)  →  Lautering (1-5)  →  Pre-wort (1-5)  →  Boiling (1-5)
    → Coolers (1-4)
    → Fermentation Lines (1-2)
    → Fermentation Tanks  – Small (48 tanks × 4 batches)
                          – Large (37 tanks × 8 batches)

All capacities and processing times are centralised in CONFIG.
Set SIM_CONFIG["verbose"] = False to suppress per-event logging.
"""

import simpy
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← edit all capacities and times here
# ═══════════════════════════════════════════════════════════════════════════
CONFIG: Dict = {
    # ── Silos ──────────────────────────────────────────────────────────────
    "silo_capacity":          100,   # max batches stored per silo
    "batch_gen_interval":       5,   # time units between batch releases

    # ── Malt Transfer Lines ────────────────────────────────────────────────
    "malt_line_slots":          3,   # concurrent batches a line can carry
    "malt_line_transfer_time":  2,   # processing time (time units)

    # ── Mills ──────────────────────────────────────────────────────────────
    "mill_slots":               2,   # concurrent batches
    "mill_processing_time":     3,

    # ── Mashing ────────────────────────────────────────────────────────────
    "mashing_slots":            1,   # one batch at a time per tank
    "mashing_time":            10,

    # ── Lautering ──────────────────────────────────────────────────────────
    "lautering_slots":          1,
    "lautering_time":           8,

    # ── Pre-wort ───────────────────────────────────────────────────────────
    "prewort_slots":            1,
    "prewort_time":             5,

    # ── Boiling ────────────────────────────────────────────────────────────
    "boiling_slots":            1,
    "boiling_time":            15,

    # ── Coolers ────────────────────────────────────────────────────────────
    "cooler_slots":             2,
    "cooling_time":             6,

    # ── Fermentation Lines ─────────────────────────────────────────────────
    "ferm_line_slots":          3,
    "ferm_line_time":           2,

    # ── Fermentation Tanks ─────────────────────────────────────────────────
    "small_tank_count":        48,
    "small_tank_capacity":      4,   # batches per tank
    "small_fermentation_time": 48,

    "large_tank_count":        37,
    "large_tank_capacity":      8,
    "large_fermentation_time": 72,

    # ── Simulation ─────────────────────────────────────────────────────────
    "sim_duration":           500,
    "num_batches":             30,
    "random_seed":             42,
    "verbose":               True,
}

# ═══════════════════════════════════════════════════════════════════════════
#  TOPOLOGY – adjacency lists
#  Each connection list is used by connect() to route batches.
# ═══════════════════════════════════════════════════════════════════════════
CONNECTIONS = {
    # silo → malt transfer line  (all-to-all; routing = random choice)
    "silo_to_malt": {
        "M": ["Cimbria", "Kunzel", "Alapros"],
        "N": ["Cimbria", "Kunzel", "Alapros"],
        "O": ["Cimbria", "Kunzel", "Alapros"],
        "P": ["Cimbria", "Kunzel", "Alapros"],
        "R": ["Cimbria", "Kunzel", "Alapros"],
    },
    # malt transfer line → mill
    "malt_to_mill": {
        "Cimbria": ["Millstar", "Powermill"],
        "Kunzel":  ["Millstar", "Powermill", "Variomill"],
        "Alapros": ["Powermill", "Variomill"],
    },
    # mill → mashing tank
    "mill_to_mashing": {
        "Millstar":  [1, 2],
        "Powermill": [2, 3, 4],
        "Variomill": [4, 5],
    },
    # mashing → lautering  (1-to-1)
    "mashing_to_lautering": {i: i for i in range(1, 6)},
    # lautering → pre-wort  (1-to-1)
    "lautering_to_prewort":  {i: i for i in range(1, 6)},
    # pre-wort → boiling    (1-to-1)
    "prewort_to_boiling":    {i: i for i in range(1, 6)},
    # boiling → cooler      (all-to-all)
    "boiling_to_cooler": {i: [1, 2, 3, 4] for i in range(1, 6)},
    # cooler → fermentation line  (all-to-all)
    "cooler_to_fermline": {i: [1, 2] for i in range(1, 5)},
    # fermentation line → tank type
    "fermline_to_tank": {
        1: ["Small", "Large"],
        2: ["Small", "Large"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Batch:
    """A single production batch that travels through the brewery."""
    batch_id:      int
    origin_silo:   str
    creation_time: float
    current_stage: str = "created"
    history:       List[Tuple[float, str]] = field(default_factory=list)
    completed:     bool = False
    tank_type:     Optional[str] = None   # "Small" or "Large"

    def advance(self, stage: str, env_now: float) -> None:
        self.current_stage = stage
        self.history.append((env_now, stage))
        if CONFIG["verbose"]:
            log.info(f"  [t={env_now:6.1f}]  Batch {self.batch_id:03d}  →  {stage}")


@dataclass
class EquipmentFlag:
    """Runtime status flags for a single piece of equipment."""
    name:               str
    active:             bool  = False   # powered on
    busy:               bool  = False   # currently processing
    batches_processed:  int   = 0
    current_batch_id:   Optional[int] = None

    def start(self, batch_id: int) -> None:
        self.busy             = True
        self.current_batch_id = batch_id
        self.batches_processed += 1

    def finish(self) -> None:
        self.busy             = False
        self.current_batch_id = None


@dataclass
class ConnectionFlag:
    """Tracks how many batches have traversed a connection."""
    from_eq:   str
    to_eq:     str
    transfers: int  = 0
    active:    bool = False

    def transfer(self) -> None:
        self.active    = True
        self.transfers += 1

    def idle(self) -> None:
        self.active = False


# ═══════════════════════════════════════════════════════════════════════════
#  BEER PRODUCTION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
class BeerProductionSystem:
    """
    Full brewery simulation built on SimPy.

    Equipment is modelled as simpy.Resource (capacity = concurrent slots).
    Flags (EquipmentFlag, ConnectionFlag) mirror real-time state and are
    updated at every acquire/release event.
    """

    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, env: simpy.Environment, cfg: Dict = CONFIG):
        self.env = env
        self.cfg = cfg
        random.seed(cfg["random_seed"])

        # Results store
        self.completed_batches: List[Batch] = []

        # ── Build equipment ────────────────────────────────────────────────
        self._build_silos()
        self._build_malt_transfer_lines()
        self._build_mills()
        self._build_mashing()
        self._build_lautering()
        self._build_prewort()
        self._build_boiling()
        self._build_coolers()
        self._build_fermentation_lines()
        self._build_fermentation_tanks()

        # ── Build connection flags ─────────────────────────────────────────
        self._build_connection_flags()

        log.info("=== BeerProductionSystem initialised ===\n")

    # ══════════════════════════════════════════════════════════════════════
    #  EQUIPMENT BUILDERS
    # ══════════════════════════════════════════════════════════════════════

    def _build_silos(self) -> None:
        """Five raw-material silos (M, N, O, P, R)."""
        names = ["M", "N", "O", "P", "R"]
        self.silos: Dict[str, simpy.Container]  = {}
        self.silo_flags: Dict[str, EquipmentFlag] = {}
        for n in names:
            self.silos[n] = simpy.Container(
                self.env, capacity=self.cfg["silo_capacity"],
                init=self.cfg["silo_capacity"]   # start full
            )
            self.silo_flags[n] = EquipmentFlag(name=n, active=True)

    def _build_malt_transfer_lines(self) -> None:
        """Three malt transfer lines (Cimbria, Kunzel, Alapros)."""
        names = ["Cimbria", "Kunzel", "Alapros"]
        self.malt_lines: Dict[str, simpy.Resource]    = {}
        self.malt_line_flags: Dict[str, EquipmentFlag] = {}
        for n in names:
            self.malt_lines[n] = simpy.Resource(
                self.env, capacity=self.cfg["malt_line_slots"]
            )
            self.malt_line_flags[n] = EquipmentFlag(name=n, active=True)

    def _build_mills(self) -> None:
        """Three mills (Millstar, Powermill, Variomill)."""
        names = ["Millstar", "Powermill", "Variomill"]
        self.mills: Dict[str, simpy.Resource]    = {}
        self.mill_flags: Dict[str, EquipmentFlag] = {}
        for n in names:
            self.mills[n] = simpy.Resource(
                self.env, capacity=self.cfg["mill_slots"]
            )
            self.mill_flags[n] = EquipmentFlag(name=n, active=True)

    def _build_mashing(self) -> None:
        """Five mashing tanks (1-5)."""
        self.mashing: Dict[int, simpy.Resource]    = {}
        self.mashing_flags: Dict[int, EquipmentFlag] = {}
        for i in range(1, 6):
            self.mashing[i] = simpy.Resource(
                self.env, capacity=self.cfg["mashing_slots"]
            )
            self.mashing_flags[i] = EquipmentFlag(name=f"Mashing-{i}", active=True)

    def _build_lautering(self) -> None:
        """Five lautering tanks (1-5)."""
        self.lautering: Dict[int, simpy.Resource]    = {}
        self.lautering_flags: Dict[int, EquipmentFlag] = {}
        for i in range(1, 6):
            self.lautering[i] = simpy.Resource(
                self.env, capacity=self.cfg["lautering_slots"]
            )
            self.lautering_flags[i] = EquipmentFlag(name=f"Lautering-{i}", active=True)

    def _build_prewort(self) -> None:
        """Five pre-wort tanks (1-5)."""
        # Named 'prewort_tanks' (not 'prewort') to avoid shadowing the prewort() method.
        self.prewort_tanks: Dict[int, simpy.Resource]    = {}
        self.prewort_flags: Dict[int, EquipmentFlag] = {}
        for i in range(1, 6):
            self.prewort_tanks[i] = simpy.Resource(
                self.env, capacity=self.cfg["prewort_slots"]
            )
            self.prewort_flags[i] = EquipmentFlag(name=f"Prewort-{i}", active=True)

    def _build_boiling(self) -> None:
        """Five boiling kettles (1-5)."""
        self.boiling: Dict[int, simpy.Resource]    = {}
        self.boiling_flags: Dict[int, EquipmentFlag] = {}
        for i in range(1, 6):
            self.boiling[i] = simpy.Resource(
                self.env, capacity=self.cfg["boiling_slots"]
            )
            self.boiling_flags[i] = EquipmentFlag(name=f"Boiling-{i}", active=True)

    def _build_coolers(self) -> None:
        """Four coolers (1-4)."""
        self.coolers: Dict[int, simpy.Resource]    = {}
        self.cooler_flags: Dict[int, EquipmentFlag] = {}
        for i in range(1, 5):
            self.coolers[i] = simpy.Resource(
                self.env, capacity=self.cfg["cooler_slots"]
            )
            self.cooler_flags[i] = EquipmentFlag(name=f"Cooler-{i}", active=True)

    def _build_fermentation_lines(self) -> None:
        """Two fermentation lines (1-2)."""
        self.ferm_lines: Dict[int, simpy.Resource]    = {}
        self.ferm_line_flags: Dict[int, EquipmentFlag] = {}
        for i in range(1, 3):
            self.ferm_lines[i] = simpy.Resource(
                self.env, capacity=self.cfg["ferm_line_slots"]
            )
            self.ferm_line_flags[i] = EquipmentFlag(name=f"FermLine-{i}", active=True)

    def _build_fermentation_tanks(self) -> None:
        """
        Small tanks : 48 tanks × 4 batches capacity.
        Large tanks : 37 tanks × 8 batches capacity.
        Modelled as a single Container per type (total capacity = count × per-tank cap).
        """
        small_total = self.cfg["small_tank_count"] * self.cfg["small_tank_capacity"]
        large_total = self.cfg["large_tank_count"] * self.cfg["large_tank_capacity"]

        self.ferm_tanks: Dict[str, simpy.Container] = {
            "Small": simpy.Container(self.env, capacity=small_total, init=0),
            "Large": simpy.Container(self.env, capacity=large_total, init=0),
        }
        self.ferm_tank_flags: Dict[str, EquipmentFlag] = {
            "Small": EquipmentFlag(name="FermTank-Small", active=True),
            "Large": EquipmentFlag(name="FermTank-Large", active=True),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  CONNECTION FLAG BUILDER
    # ══════════════════════════════════════════════════════════════════════
    def _build_connection_flags(self) -> None:
        """Create a ConnectionFlag for every directed edge in CONNECTIONS."""
        self.conn_flags: Dict[str, ConnectionFlag] = {}

        def _add(src, dst):
            key = f"{src}→{dst}"
            self.conn_flags[key] = ConnectionFlag(from_eq=str(src), to_eq=str(dst))

        for src, dsts in CONNECTIONS["silo_to_malt"].items():
            for d in dsts: _add(src, d)
        for src, dsts in CONNECTIONS["malt_to_mill"].items():
            for d in dsts: _add(src, d)
        for src, dsts in CONNECTIONS["mill_to_mashing"].items():
            for d in dsts: _add(src, d)
        for s, d in CONNECTIONS["mashing_to_lautering"].items():  _add(f"Mashing-{s}", f"Lautering-{d}")
        for s, d in CONNECTIONS["lautering_to_prewort"].items():  _add(f"Lautering-{s}", f"Prewort-{d}")
        for s, d in CONNECTIONS["prewort_to_boiling"].items():    _add(f"Prewort-{s}", f"Boiling-{d}")
        for src, dsts in CONNECTIONS["boiling_to_cooler"].items():
            for d in dsts: _add(f"Boiling-{src}", f"Cooler-{d}")
        for src, dsts in CONNECTIONS["cooler_to_fermline"].items():
            for d in dsts: _add(f"Cooler-{src}", f"FermLine-{d}")
        for src, dsts in CONNECTIONS["fermline_to_tank"].items():
            for d in dsts: _add(f"FermLine-{src}", f"FermTank-{d}")

    # ══════════════════════════════════════════════════════════════════════
    #  CONNECTION HELPER
    # ══════════════════════════════════════════════════════════════════════
    def connect(self, from_eq: str, to_eq: str) -> None:
        """Record that a batch just traversed the edge from_eq → to_eq."""
        key = f"{from_eq}→{to_eq}"
        if key in self.conn_flags:
            self.conn_flags[key].transfer()

    # ══════════════════════════════════════════════════════════════════════
    #  EQUIPMENT PROCESS FUNCTIONS
    # ══════════════════════════════════════════════════════════════════════

    def silo_release(self, batch: Batch, silo_name: str):
        """Withdraw one batch-unit from a silo."""
        yield self.silos[silo_name].get(1)
        flag = self.silo_flags[silo_name]
        flag.start(batch.batch_id)
        batch.advance(f"released-from-silo-{silo_name}", self.env.now)
        flag.finish()

    def malt_transfer_line(self, batch: Batch, line_name: str):
        """Acquire a malt transfer line slot and transfer the batch."""
        with self.malt_lines[line_name].request() as req:
            yield req
            flag = self.malt_line_flags[line_name]
            flag.start(batch.batch_id)
            batch.advance(f"in-MaltLine-{line_name}", self.env.now)
            yield self.env.timeout(self.cfg["malt_line_transfer_time"])
            flag.finish()

    def mill(self, batch: Batch, mill_name: str):
        """Grind malt in the specified mill."""
        with self.mills[mill_name].request() as req:
            yield req
            flag = self.mill_flags[mill_name]
            flag.start(batch.batch_id)
            batch.advance(f"in-Mill-{mill_name}", self.env.now)
            yield self.env.timeout(self.cfg["mill_processing_time"])
            flag.finish()

    def mash(self, batch: Batch, tank: int):
        """Mashing step."""
        with self.mashing[tank].request() as req:
            yield req
            flag = self.mashing_flags[tank]
            flag.start(batch.batch_id)
            batch.advance(f"in-Mashing-{tank}", self.env.now)
            yield self.env.timeout(self.cfg["mashing_time"])
            flag.finish()

    def lauter(self, batch: Batch, tank: int):
        """Lautering step."""
        with self.lautering[tank].request() as req:
            yield req
            flag = self.lautering_flags[tank]
            flag.start(batch.batch_id)
            batch.advance(f"in-Lautering-{tank}", self.env.now)
            yield self.env.timeout(self.cfg["lautering_time"])
            flag.finish()

    def prewort(self, batch: Batch, tank: int):
        """Pre-wort collection step."""
        with self.prewort_tanks[tank].request() as req:
            yield req
            flag = self.prewort_flags[tank]
            flag.start(batch.batch_id)
            batch.advance(f"in-Prewort-{tank}", self.env.now)
            yield self.env.timeout(self.cfg["prewort_time"])
            flag.finish()

    def boil(self, batch: Batch, tank: int):
        """Boiling step."""
        with self.boiling[tank].request() as req:
            yield req
            flag = self.boiling_flags[tank]
            flag.start(batch.batch_id)
            batch.advance(f"in-Boiling-{tank}", self.env.now)
            yield self.env.timeout(self.cfg["boiling_time"])
            flag.finish()

    def cool(self, batch: Batch, cooler: int):
        """Cooling step."""
        with self.coolers[cooler].request() as req:
            yield req
            flag = self.cooler_flags[cooler]
            flag.start(batch.batch_id)
            batch.advance(f"in-Cooler-{cooler}", self.env.now)
            yield self.env.timeout(self.cfg["cooling_time"])
            flag.finish()

    def fermentation_line(self, batch: Batch, line: int):
        """Transfer through a fermentation line."""
        with self.ferm_lines[line].request() as req:
            yield req
            flag = self.ferm_line_flags[line]
            flag.start(batch.batch_id)
            batch.advance(f"in-FermLine-{line}", self.env.now)
            yield self.env.timeout(self.cfg["ferm_line_time"])
            flag.finish()

    def fermentation_tank(self, batch: Batch, tank_type: str):
        """
        Place batch in a fermentation tank.
        Container.put() blocks when the tank group is full.
        """
        yield self.ferm_tanks[tank_type].put(1)
        flag = self.ferm_tank_flags[tank_type]
        flag.start(batch.batch_id)
        ferm_time = (
            self.cfg["small_fermentation_time"] if tank_type == "Small"
            else self.cfg["large_fermentation_time"]
        )
        batch.tank_type = tank_type
        batch.advance(f"fermenting-in-{tank_type}-tank", self.env.now)
        yield self.env.timeout(ferm_time)

        # Release the tank slot
        yield self.ferm_tanks[tank_type].get(1)
        flag.finish()
        batch.advance(f"fermentation-complete", self.env.now)
        batch.completed = True
        self.completed_batches.append(batch)

    # ══════════════════════════════════════════════════════════════════════
    #  ROUTING HELPERS  (pick least-loaded resource from allowed targets)
    # ══════════════════════════════════════════════════════════════════════
    def _pick_resource(self, resource_dict, allowed_keys) -> any:
        """Return the key of the least-loaded resource among allowed_keys."""
        return min(allowed_keys,
                   key=lambda k: resource_dict[k].count)   # count = in-use slots

    # ══════════════════════════════════════════════════════════════════════
    #  BATCH PIPELINE  (main generator)
    # ══════════════════════════════════════════════════════════════════════
    def batch_pipeline(self, batch: Batch):
        """
        Drive one batch through the entire production pipeline.
        Each stage calls the appropriate equipment function and records
        the connection it just traversed.
        """
        silo = batch.origin_silo

        # 1. Silo release ──────────────────────────────────────────────────
        yield from self.silo_release(batch, silo)

        # 2. Malt Transfer Line ────────────────────────────────────────────
        line_name = self._pick_resource(
            self.malt_lines, CONNECTIONS["silo_to_malt"][silo]
        )
        self.connect(silo, line_name)
        yield from self.malt_transfer_line(batch, line_name)

        # 3. Mill ──────────────────────────────────────────────────────────
        mill_name = self._pick_resource(
            self.mills, CONNECTIONS["malt_to_mill"][line_name]
        )
        self.connect(line_name, mill_name)
        yield from self.mill(batch, mill_name)

        # 4. Mashing ───────────────────────────────────────────────────────
        mashing_id = self._pick_resource(
            self.mashing, CONNECTIONS["mill_to_mashing"][mill_name]
        )
        self.connect(mill_name, f"Mashing-{mashing_id}")
        yield from self.mash(batch, mashing_id)

        # 5. Lautering (follows mashing tank index) ──────────────────────
        lautering_id = CONNECTIONS["mashing_to_lautering"][mashing_id]
        self.connect(f"Mashing-{mashing_id}", f"Lautering-{lautering_id}")
        yield from self.lauter(batch, lautering_id)

        # 6. Pre-wort ──────────────────────────────────────────────────────
        prewort_id = CONNECTIONS["lautering_to_prewort"][lautering_id]
        self.connect(f"Lautering-{lautering_id}", f"Prewort-{prewort_id}")
        yield from self.prewort(batch, prewort_id)

        # 7. Boiling ───────────────────────────────────────────────────────
        boiling_id = CONNECTIONS["prewort_to_boiling"][prewort_id]
        self.connect(f"Prewort-{prewort_id}", f"Boiling-{boiling_id}")
        yield from self.boil(batch, boiling_id)

        # 8. Cooler ────────────────────────────────────────────────────────
        cooler_id = self._pick_resource(
            self.coolers, CONNECTIONS["boiling_to_cooler"][boiling_id]
        )
        self.connect(f"Boiling-{boiling_id}", f"Cooler-{cooler_id}")
        yield from self.cool(batch, cooler_id)

        # 9. Fermentation Line ─────────────────────────────────────────────
        ferm_line_id = self._pick_resource(
            self.ferm_lines, CONNECTIONS["cooler_to_fermline"][cooler_id]
        )
        self.connect(f"Cooler-{cooler_id}", f"FermLine-{ferm_line_id}")
        yield from self.fermentation_line(batch, ferm_line_id)

        # 10. Fermentation Tank (prefer Small; fall back to Large) ──────────
        tank_type = self._pick_tank()
        self.connect(f"FermLine-{ferm_line_id}", f"FermTank-{tank_type}")
        yield from self.fermentation_tank(batch, tank_type)

    def _pick_tank(self) -> str:
        """
        Route to Small tank if space is available, otherwise Large.
        Space available = current level < capacity.
        """
        small = self.ferm_tanks["Small"]
        if small.level < small.capacity:
            return "Small"
        return "Large"

    # ══════════════════════════════════════════════════════════════════════
    #  BATCH GENERATOR  (source process)
    # ══════════════════════════════════════════════════════════════════════
    def batch_source(self):
        """Generate batches at regular intervals, cycling through silos."""
        silos = list(self.silos.keys())
        for batch_id in range(1, self.cfg["num_batches"] + 1):
            silo = silos[(batch_id - 1) % len(silos)]
            batch = Batch(
                batch_id=batch_id,
                origin_silo=silo,
                creation_time=self.env.now,
            )
            log.info(f"\n>>> Batch {batch_id:03d} created at silo {silo}  "
                     f"(t={self.env.now:.1f})")
            self.env.process(self.batch_pipeline(batch))
            yield self.env.timeout(self.cfg["batch_gen_interval"])

    # ══════════════════════════════════════════════════════════════════════
    #  RUN
    # ══════════════════════════════════════════════════════════════════════
    def run(self):
        """Start the source process and advance the simulation clock."""
        self.env.process(self.batch_source())
        self.env.run(until=self.cfg["sim_duration"])

    # ══════════════════════════════════════════════════════════════════════
    #  REPORTING
    # ══════════════════════════════════════════════════════════════════════
    def print_summary(self):
        total   = self.cfg["num_batches"]
        done    = len(self.completed_batches)
        small_c = sum(1 for b in self.completed_batches if b.tank_type == "Small")
        large_c = sum(1 for b in self.completed_batches if b.tank_type == "Large")

        print("\n" + "═" * 60)
        print("  SIMULATION SUMMARY")
        print("═" * 60)
        print(f"  Simulation time    : {self.cfg['sim_duration']} time units")
        print(f"  Batches generated  : {total}")
        print(f"  Batches completed  : {done}  "
              f"(Small: {small_c}  Large: {large_c})")

        print("\n  Equipment utilisation (batches processed):")
        for name, flag in {**self.silo_flags,
                           **self.malt_line_flags,
                           **self.mill_flags}.items():
            print(f"    {name:<14}: {flag.batches_processed}")

        print("\n  Connection transfers (top 10):")
        sorted_conns = sorted(self.conn_flags.values(),
                              key=lambda c: c.transfers, reverse=True)
        for cf in sorted_conns[:10]:
            if cf.transfers > 0:
                print(f"    {cf.from_eq:<18} → {cf.to_eq:<18}: {cf.transfers}")

        print("\n  Fermentation tanks (current level / capacity):")
        for ttype, cont in self.ferm_tanks.items():
            print(f"    {ttype:<8}: {cont.level:.0f} / {cont.capacity:.0f}")

        if self.completed_batches:
            lead_times = [
                b.history[-1][0] - b.creation_time
                for b in self.completed_batches
            ]
            print(f"\n  Lead time  min/avg/max : "
                  f"{min(lead_times):.1f} / "
                  f"{sum(lead_times)/len(lead_times):.1f} / "
                  f"{max(lead_times):.1f} time units")
        print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    env    = simpy.Environment()
    system = BeerProductionSystem(env, CONFIG)
    system.run()
    system.print_summary()