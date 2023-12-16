import modal
from PIL import Image

MODEL_VERSION = 7
LOCAL = False
API_KEY = 'IjpOAHZvfhOIKz5Z.nSNpsCdomwyvjmYZ8utAUd1HE8up18q1eoi4oA41wMMHJlSbVK8mZNUJPbOjs1li'

if LOCAL == False:
   hopsworks_image = modal.Image.debian_slim(python_version='3.9').pip_install(["gradio", "requests", "hopsworks", "joblib", "pandas", "scikit-learn==1.1.1"])
   stub = modal.Stub("wine_prediction_user_interface")
   @stub.function(image=hopsworks_image, secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import gradio as gr
    import requests
    import hopsworks
    import joblib
    import pandas as pd

    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=MODEL_VERSION)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    print("Model downloaded")

    def wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
            total_sulfar_dioxide, density, ph, sulphates, alcohol, color):
        print("Calling function")
    #     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
        df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfar_dioxide, density, ph, sulphates, alcohol, color]], 
                        columns=['fixed_acidity', 'volatile_acidity', 'citric_aicd', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                            'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'color'])
        print("Predicting")
        print(df)
        # 'res' is a list of predictions returned as the label.
        res = model.predict(df)
        # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
        # the first element.
    #     print("Res: {0}").format(res)
        print(res)
        return res
            
    demo = gr.Interface(
        fn=wine,
        title="Wine Quality Predictive Analytics",
        description="Experiment with several parameters to predict the quality of wine.",
        allow_flagging="never",
        inputs=[
            gr.Number(value=1.0, label="fixed_acidity"),
            gr.Number(value=1.0, label="volatile_acidity"),
            gr.Number(value=1.0, label="citric_aicd"),
            gr.Number(value=1.0, label="residual_sugar"),
            gr.Number(value=1.0, label="chlorides"),
            gr.Number(value=1.0, label="free_sulfur_dioxide"),
            gr.Number(value=1.0, label="total_sulfur_dioxide"),
            gr.Number(value=1.0, label="density"),
            gr.Number(value=1.0, label="ph"),
            gr.Number(value=1.0, label="sulphates"),
            gr.Number(value=1.0, label="alcohol"),
            gr.Number(value=1.0, label="color (0:red, 1:white)"),
            ],
        outputs=gr.Number(label="predicted quality"))

    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        with stub.run():
            f.remote()