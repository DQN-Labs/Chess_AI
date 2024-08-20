import collections
import random
import sys

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import gtp
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "human",

    # Run an external program that speaks the Go Text Protocol.
    # Requires the gtp_path flag.
    "gtp",

    # Run an alpha_zero checkpoint with MCTS. Uses the specified UCT/sims.
    # Requires the az_path flag.
    "az"
]

flags.DEFINE_string("game", "chess", "Name of the game.")
flags.DEFINE_enum("player1", "mcts", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "human", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_string("gtp_path", None, "Where to find a binary for gtp.")
flags.DEFINE_multi_string("gtp_cmd", [], "GTP commands to run at init.")
flags.DEFINE_string("az_path", None,
                    "Path to an alpha_zero checkpoint. Needed by an az player.")
flags.DEFINE_integer("uct_c", 3, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 70, "How many simulations to run.")
flags.DEFINE_integer("num_games", 10, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("random_first", False, "Play the first move randomly.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", True, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
  if not FLAGS.quiet:
    print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
  """Initializes a bot by type."""
  rng = np.random.RandomState(FLAGS.seed)
  if bot_type == "mcts":
    evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "az":
    model = az_model.Model.from_checkpoint(FLAGS.az_path)
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)
    return mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=rng,
        child_selection_fn=mcts.SearchNode.puct_value,
        solve=FLAGS.solve,
        verbose=FLAGS.verbose)
  if bot_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  if bot_type == "human":
    return human.HumanBot()
  if bot_type == "gtp":
    bot = gtp.GTPBot(game, FLAGS.gtp_path)
    for cmd in FLAGS.gtp_cmd:
      bot.gtp_cmd(cmd)
    return bot
  raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action(state, action_str):
  for action in state.legal_actions():
    if action_str == state.action_to_string(state.current_player(), action):
      return action
  return None


def _print_chess_board(state):
    """Prints the chess board in a human-readable format."""
    board = state.to_string()  # Convert the state to a string representation
    # Example formatting: You might need to parse the `board` string for better display
    lines = board.split('\n')
    for line in lines:
        print(line)
    print("\nLegal moves: {}".format([state.action_to_string(state.current_player(), a) for a in state.legal_actions()]))

def _play_game(game, bots, initial_actions):
    """Plays one game."""
    state = game.new_initial_state()
    _print_chess_board(state)

    history = []

    if FLAGS.random_first:
        assert not initial_actions
        initial_actions = [state.action_to_string(
            state.current_player(), random.choice(state.legal_actions()))]

    for action_str in initial_actions:
        action = _get_action(state, action_str)
        if action is None:
            sys.exit("Invalid action: {}".format(action_str))

        history.append(action_str)
        for bot in bots:
            bot.inform_action(state, state.current_player(), action)
        state.apply_action(action)
        _print_chess_board(state)

    while not state.is_terminal():
        current_player = state.current_player()
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            _print_chess_board(state)
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            action_str = state.action_to_string(current_player, action)
            _print_chess_board(state)
        elif state.is_simultaneous_node():
            raise ValueError("Game cannot have simultaneous nodes.")
        else:
            bot = bots[current_player]
            if isinstance(bot, human.HumanBot):
                _print_chess_board(state)
                action_str = input("Enter your move (e.g., 'e2e4'): ")
                action = _get_action(state, action_str)
                if action is None:
                    sys.exit("Invalid move: {}".format(action_str))
            else:
                action = bot.step(state)
                action_str = state.action_to_string(current_player, action)
                _print_chess_board(state)

        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
        history.append(action_str)
        state.apply_action(action)

        _print_chess_board(state)

    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)), ", Game actions:",
          " ".join(history))

    for bot in bots:
        bot.restart()

    return returns, history



def main(argv):
    game = pyspiel.load_game(FLAGS.game)
    if game.num_players() > 2:
        sys.exit("This game requires more players than the example can handle.")

    # Initialize bots
    bots = [
        _init_bot(FLAGS.player1, game, 0),
        _init_bot(FLAGS.player2, game, 1),
    ]

    # Track the number of games and wins
    histories = collections.defaultdict(int)
    overall_returns = [0, 0]
    overall_wins = [0, 0]
    game_num = 0

    try:
        for game_num in range(FLAGS.num_games):
            # After 100 games, replace one bot with a human bot
            if game_num == 2:
                bots[1] = human.HumanBot()
                print("Now playing against a human.")

            returns, history = _play_game(game, bots, argv[1:])
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
    except (KeyboardInterrupt, EOFError):
        game_num -= 1
        print("Caught a KeyboardInterrupt, stopping early.")

    print("Number of games played:", game_num + 1)
    print("Number of distinct games played:", len(histories))
    print("Players:", FLAGS.player1, FLAGS.player2)
    print("Overall wins", overall_wins)
    print("Overall returns", overall_returns)


if __name__ == "__main__":
    app.run(main)


