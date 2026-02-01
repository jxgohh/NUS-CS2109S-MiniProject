Code for Mini Project done during undertaking of Introduction to AI and Machine Learning.
Notebook file grid-universe.ipynb contains documentation for the maze while the other python files contain my solutions.


## Task and Submission (Copied over task specification)

Your task is to develop an agent that plays Grid Universe effectively.

Implement the `Agent` class (see [AI Agent](#AI-Agent)) with your logic.

### Submission Details

#### Agent
Submit your `Agent` code to Coursemology.

- **Tasks:** To make the grand objective manageable, we provide three tasks (see [Tasks](#Tasks)), each isolates a subset of challenges, and one final task that combines everything.
  - Task 1 [12 marks]: Structured state representation; plaintext objective.
  - Task 2 [2 marks]: Structured state representation; ciphertext objective.
  - Task 3 [12 marks]: Image observation + supplemental structured data; plaintext objective.
  - Task 4 [4 marks]: Everything: multiple gameplay mechanics; image and level observation; ciphertext objective (full challenge).

- **Test Cases:** Each task includes multiple test cases. Each test case contains many level instances that may differ in mechanics, size, and complexity.

You may reuse the same agent for all tasks or adapt it per task.

### Time Limit
The time limit for each **level instance** is given below:
- Task 1, 2, 3: 25 seconds
- Task 4: 100 seconds

## Grid Universe

### Overview

Grid Universe is, at its core, a grid‑world environment enriched with varied mechanics. A level is an M×N grid (positive integers M, N). Each tile can contain multiple objects (floor, agent, portal, key, door, enemy, etc.), each with distinct behavior. The goal depends on the level’s objective: one level may require only reaching an exit; another may require collecting specified items before exiting.

Please refer to the **Grid Universe tutorial notebook** for more details.

> It is highly recommended (we would even say mandatory!) to explore the Grid Universe tutorial notebook from the beginning until the end, especially if you are lost.

**Additional resources:**

- See the [Player Guide](https://rizkiarm.github.io/grid-universe/guides/player_guide/) for a friendly introduction.

- For code and full documentation, visit the [GitHub repository](https://github.com/rizkiarm/grid-universe/) and the [API docs](https://rizkiarm.github.io/grid-universe/).

- For playing the game yourself, try the [Streamlit Demo](https://grid-universe.streamlit.app/).

### Constraints (Project Scope)
To keep the mini project tractable, we enforce the following constraints:

- Objectives are restricted to two types: `exit_objective_fn` and `default_objective_fn`.
- All object movements are deterministic and move at most one tile per step.
- The agent moves one tile per step, except under the speed (Boots) power‑up, which doubles movement to two tiles per step.
- Each entity type has a fixed, known cost, damage, or reward (cost = –reward) where applicable:
  - Coins: reward 5.
  - Cores (required items): reward 0.
  - Floor tiles: cost 3 (the action cost is the cost of the tile you stand on after the action, except the exit tile which costs 0—an action ending on the exit costs 0).
  - Hazards: damage 2.
  - Enemies: damage 1.
- Boxes and enemies can only move orthogonally and bounce. No diagonal movement or complicated AI.
- At most one key–door pair and one portal pair per level.
- Power‑up durations are fixed:
  - Boots (Speed): 5 steps.
  - Ghost (Phasing): 5 steps.
  - Shield: 5 uses.
- Agent health may vary per level.

### Objective and Challenges
Your overarching objective is to build an agent that plays well across diverse levels. Key challenges:

- **Planning:** Navigate mazes; potentially avoid enemies and hazards; open doors; use portals; sequence actions to minimize total cost / maximize total reward.
- **Image Interpretation:** The principal observation is a rendered top‑down RGBA image. You must transform it into a representation suitable for search / planning.
- **Ciphertext Objectives:** Some levels encode the objective as a fixed‑length ciphertext. Decoding it is essential; misunderstanding leads to wasted actions (e.g., collecting unnecessary cores or omitting required ones).

We decompose the overarching problem into tasks to let you focus on one challenge at a time (see [Tasks](#Tasks)).

### State Representation
The form of state depends on the task. Tasks 1 and 2 expose a structured `Level` object. Task 3 and 4 provides an RGB image plus supplementary structured information.

#### Level Object
An instance of class `Level` with attributes:

- `width: int` – grid width.
- `height: int` – grid height.
- `move_fn: function` – movement rules.
- `objective_fn: function` – objective function (redacted and replaced by a dummy in Task 2 and 4).
- `grid: List[List[EntitySpec]]` – height×width matrix; each cell is a list of `EntitySpec`.
- `score: int` – current score.
- `message: str` – ciphertext (if objective redacted) to decode.
- `win: bool`, `lose: bool` – termination flags.
- `turn: int` – current turn count.
- `turn_limit: int` – maximum turns (if any).
- `seed: int` – random seed.

`EntitySpec` describes an entity in a cell:
- Properties (components), e.g., `agent`, `blocking`, etc.
- `inventory: list` – carried items.
- `status: list` – active status effects / power‑ups.

See the Grid Universe tutorial notebook for a gentle start.

**API references:**
- [Level](https://rizkiarm.github.io/grid-universe/reference/api/#levels-authoring)
- [Components](https://rizkiarm.github.io/grid-universe/reference/api/#components)

#### Image + Extra Information

The observation is an `Observation` dictionary defined in `grid_universe.gym_env` with the following fields:

- `image`: Top‑down RGBA rendering of the current level state (shape H × W × 4, dtype `uint8`). Each (logical) grid cell is visualized; when multiple entities share a tile, their sprites are composited according to appearance priority rules.
- `info`:
  - `agent`:
    - `health`:
      - `health: int` – Current hit points after the last resolved action (cannot exceed `max_health`). 
      - `max_health: int` – Maximum possible hit points for this level configuration.
    - `effects: List[EffectEntry]` – Active temporary effects / power‑ups (e.g., speed, phasing, shield), each usually carrying remaining duration or uses.
    - `inventory: List[InventoryItem]` – Items the agent currently holds (e.g., key).
  - `status`:
    - `score: int` – Accumulated reward so far (sum of rewards minus costs).
    - `phase: str` – Game state: `"ongoing"`, `"win"`, or `"lose"`.
    - `turn: int` – Number of turns elapsed.
  - `config`:
    - `move_fn: str` – Identifier/name of the movement rule used to update entities each step.
    - `objective_fn: str` – Identifier/name of the objective function; This will be redacted in Task 2 and 4.
    - `seed: int` – Random seed used for procedural aspects (−1 if no seed / non‑deterministic origin provided).
    - `width: int` – Level width in tiles.
    - `height: int` – Level height in tiles.
    - `turn_limit: int` – Maximum allowed turns before forced termination (−1 signifies no limit).
  - `message: str` – ciphertext (if objective redacted) to decode.

Multiple objects can occupy the same tile. Rendering layers them by [priority rules](https://rizkiarm.github.io/grid-universe/guides/authoring_levels/#appearance-priority-and-layering)

For an example observation, see the [Grid Universe demo](http://grid-universe.streamlit.app).

### Actions

Actions use the `Action` enum defined in `grid_universe.gym_env`:

- Movement: `UP`, `DOWN`, `LEFT`, `RIGHT`
- `USE_KEY`: use a key in inventory (if any)
- `PICK_UP`: collect all objects in the cell with the `Collectible` component
- `WAIT`: no operation

> Note that the `Action` enum here is distinct from the (same-named) `Action` enum defined in `grid_universe.actions`, which is utilized by the engine.

For more details regarding `Observation` and `Action`, check out the [docs](https://rizkiarm.github.io/grid-universe/guides/gym_env/#observation-and-action-spaces).

### Task 1 – Structured + Plaintext Objective

Focus: Core environment understanding & planning.

Observation: Fully structured (no raw image). Objective text is plaintext.
### Task 2 – Structured + Ciphertext Objective
Focus: Objective decoding + integration with planner.

Differences from Task 1:
- The objective function is replaced with a dummy (redacted) function. The ciphertext objective string is given in `state.message`.
- Environment representation remains structured.

In this notebook, we use the ciphertext objective data in the path specified by the `CIPHERTEXT_PATH` variable which is set to `data/ciphertext_objective.csv`. You can change this to use different file.

> In the Coursemology private test cases, we will be using different ciphertext objective data which is sampled from the same ground truth. Your decoder must generalize well.
### Task 3 – Image + Structured Supplemental Data
Focus: Perception + integration with planner.

Observation:

- Primary: Image (rendered grid) requiring visual parsing.
- Supplemental: Limited structured fields (e.g., agent health, inventory, status).
- Objective: plaintext.

Challenges:

1. Visual Parsing: Convert image to structured state representation such as `Level` object.
2. Planning: Works on reconstructed grid, tolerates occasional misclassification.
3. Performance: Keep per‑frame processing small if possible.

Recommended Steps:

1. Integrate Task 1 planner.
2. Prototype image → structured state on a single level; generalize.
3. Optimize & prune unused code/weights.

In this notebook, we use the assets in the path specified by the `ASSET_PATH` variable which is set to `data/assets`. You can change this to use different assets.

> In the Coursemology private test cases, we will be using different asset image data which is sampled from the same ground truth. Your parser must generalize well.
### Task 4 – Capstone

In this task, we combine multiple gameplay mechanics in one level. This is the final frontier of the mini project!

We will consider both `Level` and `Observation` observations and ciphertext objective.

Note: your agent needs to detect plaintext vs ciphertext objective. In `Observation` observation, ciphertext objective is given in `state["info"]["message"]` if the plaintext objective function in `state["info"]["config"]["objective_fn"]` is `<REDACTED>`.
