import argparse


parser = argparse.ArgumentParser(description='proposed model')
parser.add_argument("--dataset", type=str, default='ACLED_MUSE_7COUNTRY',  help="dataset to choose" )
parser.add_argument("--device-id", type=int, default=0,help="device id to use" )
parser.add_argument("--seed", type=int, default=40, help="random seed")
parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
parser.add_argument("--dropout", type=float, default=0.2,help="dropout probability")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight-decay", type=float, default=0.00001, help="learning rate weight_decay")
parser.add_argument("--alpha", type=float, default=0, help="learning rate weight_decay")
parser.add_argument("--use-bias", type=int, default=0, help="learning rate weight_decay")

parser.add_argument("--freq-info", type=int, default=0, help="use freq info or not")
parser.add_argument("--train-history-len", type=int, default=2, help="history length")
# parser.add_argument("--test-history-len", type=int, default=3, help="history length")
parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder ")
parser.add_argument("--hidden-dropout", type=float, default=0.2,  help="hidden dropout for decoder")
parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")

parser.add_argument("--max-epochs", type=int, default=20, help="number of maximum training epochs on each time step")
parser.add_argument("--valid-step", type=int, default=2, help="valid gap")
parser.add_argument("--mode", type=str, default='pretrain', help="train mode or test mode", choices=['pretrain', 'train','gene_train', 'test'])
# parser.add_argument("--verse", type=int, default=0, help="window size change or not")
parser.add_argument("--filter", type=int, default=0, help="filter metrics or not")
parser.add_argument("--mechanism-feature-path", type=str, default='', help="optional weekly/monthly mechanism feature csv")
parser.add_argument("--mechanism-lag-only", type=int, default=1, help="use prediction-safe lagged mechanism features only")
parser.add_argument("--mechanism-feature-columns", nargs="*", default=[],
                    help="optional explicit mechanism feature columns; empty uses all eligible columns")
parser.add_argument("--explanation-export-dir", type=str, default='',
                    help="optional directory for top-fact and branch-contribution exports")
parser.add_argument("--explanation-top-k", type=int, default=10,
                    help="top predicted facts exported per time window")
parser.add_argument("--core-protection", type=int, default=0,
                    help="upweight China/Taiwan-related dynamic training queries")
parser.add_argument("--core-replay-weight", type=float, default=3.0,
                    help="loss weight for China/Taiwan-related queries when core protection is enabled")
parser.add_argument("--use-diffusion", type=int, default=0, help="enable latent scenario diffusion generator")
parser.add_argument("--diffusion-steps", type=int, default=8, help="denoising steps for scenario diffusion")
parser.add_argument("--diffusion-samples", type=int, default=4, help="number of sampled future scenarios during evaluation")
parser.add_argument("--diffusion-noise-scale", type=float, default=0.15, help="noise scale for latent scenario diffusion")
parser.add_argument("--diffusion-loss-weight", type=float, default=0.05, help="weight of diffusion reconstruction loss")
parser.add_argument("--diffusion-use-mean-score", type=int, default=0, help="use sampled scenario mean score for ranking")
parser.add_argument("--diffusion-export-path", type=str, default='', help="optional csv path for diffusion uncertainty summaries")
parser.add_argument("--use-query-aware-context", type=int, default=0,
                    help="enable query-specific attention over mechanism context features")
parser.add_argument("--historical-contrastive-weight", type=float, default=0.0,
                    help="auxiliary weight for predicting whether the true tail is historical")
parser.add_argument("--rule-branch-weight", type=float, default=0.0,
                    help="mixture weight for the learned mechanism-rule branch")
parser.add_argument("--checkpoint-tag", type=str, default='',
                    help="optional suffix tag to keep sensitivity checkpoints from overwriting each other")
args = parser.parse_args()
