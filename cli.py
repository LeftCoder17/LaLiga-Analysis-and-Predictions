#!/usr/bin/env python
import logging
import argparse
from datetime import datetime

import settings
from quiniela import models, io


def parse_seasons(value):
    if value == "all":
        return "all"
    seasons = []
    for chunk in value.split(","):
        if ":" in chunk:
            try:
                start, end = map(int, chunk.split(":"))
                assert start < end
            except Exception:
                raise argparse.ArgumentTypeError(f"Unexpected format for seasons {value}")
            for i in range(start, end):
                seasons.append(f"{i}-{i+1}")
        else:
            try:
                start, end = map(int, chunk.split("-"))
                assert start == end - 1
            except Exception:
                raise argparse.ArgumentTypeError(f"Unexpected format for seasons {value}")
            seasons.append(chunk)
    return seasons


parser = argparse.ArgumentParser()
task_subparser = parser.add_subparsers(help='Task to perform', dest='task')
train_parser = task_subparser.add_parser("train")
train_parser.add_argument(
    "--training_seasons",
    default="all",
    type=parse_seasons,
    help="Seasons to use for training. Write them separated with ',' or use range with ':'. "
        "For instance, '2004:2006' is the same as '2004-2005,2005-2006'. "
        "Use 'all' to train with all seasons available in database.",
)
train_parser.add_argument(
    "--model_name",
    default="my_quiniela.model",
    help="The name to save the model with.",
)
predict_parser = task_subparser.add_parser("predict")
predict_parser.add_argument(
    "season",
    help="Season to predict",
)
predict_parser.add_argument(
    "division",
    type=int,
    choices=[1, 2],
    help="Division to predict (either 1 or 2)",
)
predict_parser.add_argument(
    "matchday",
    type=int,
    help="Matchday to predict",
)
predict_parser.add_argument(
    "--model_name",
    default="my_quiniela.model",
    help="The name of the model you want to use.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(
        filename=settings.LOGS_PATH / f"{args.task}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
    )
    if args.task == "train":
        logging.info(f"Training LaQuiniela model with seasons {args.training_seasons}")
        seasons = args.training_seasons
        model = models.QuinielaModel()
        training_dataset = io.preparing_training_dataset(seasons)
        features = ['season', 'division', 'matchday', 'weekday', 'time',
                    'team', 'rival', 'last_5_results_local', 'last_5_results_rival']
        target = "winner"
        X_train, y_train = training_dataset[features], training_dataset[target]
        model.train(X_train, y_train)
        model.save(settings.MODELS_PATH / args.model_name)
        print(f"Model succesfully trained and saved in {settings.MODELS_PATH / args.model_name}")
    if args.task == "predict":
        logging.info(f"Predicting matchday {args.matchday} in season {args.season}, division {args.division}")
        model = models.QuinielaModel.load(settings.MODELS_PATH / args.model_name)
        predicting_dataset = io.preparing_predicting_dataset(args.season, args.division, args.matchday)
        features = ['season', 'division', 'matchday', 'weekday', 'time',
                    'team', 'rival', 'last_5_results_local', 'last_5_results_rival']
        target = "winner"
        predict_data = predicting_dataset[features]
        predictions = model.predict(predict_data)
        probabilities = model.predict(predict_data, True)
        predict_data['pred'] = predictions
        predict_data["prob_1"] = probabilities[:, 0]
        predict_data["prob_X"] = probabilities[:, 1]
        predict_data["prob_2"] = probabilities[:, 2]
        team_names, team_ids = io.encoder_teams(args.season)
        predict_data[['team', 'rival']] = predict_data[['team', 'rival']].replace(team_ids, team_names)
        predict_data['pred'] = predict_data['pred'].replace([0, 1, 2], [1, 'X', 2])

        print(f"Matchday {args.matchday} - LaLiga - Division {args.division} - Season {args.season}")
        print("=" * 70)
        for _, row in predict_data.iterrows():
            print(f"{row['team']:^30s} vs {row['rival']:^30s} --> Pred: {row['pred']} | Prob: 1: {row['prob_1']:.2f}, X: {row['prob_X']:.2f}, 2: {row['prob_2']:.2f}")
            
        io.save_predictions(predict_data)
