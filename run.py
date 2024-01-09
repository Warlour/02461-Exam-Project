from mainclass import *
name = "Best model, best parameters, weighted"
modelhandler = ModelHandler(
    model =        EmotionRecognizerV2,
    weighted =     True,
    batch_size =   64,
    epochs =       100,
    gamma =        0.5,
    min_lr =       0,
    momentum =     0.9,
    name=name,

    start_lr =     0.001,
    weight_decay = 0.0001
)

#modelhandler.load_model("models/New tests/Test 5/2024-1-8 17_59_55 l1.7765 a0.2 CrossEntropyLoss-Adam-None_lowest_loss Test 5.pt")
modelhandler.train(stoppage=True)
modelhandler.test()
modelhandler.save_model(f"models/{name}", save_lowest=True)
modelhandler.save_excel(f"models/{name}")
modelhandler.plot_trainvstestloss(save_path=f"models/{name}", display_plot=False)
modelhandler.plot_confusionmatrix(save_path=f"models/{name}", display_plot=False)