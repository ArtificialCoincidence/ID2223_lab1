import modal
import gradio as gr
from PIL import Image

LOCAL = True

if LOCAL==False:
   stub = modal.Stub("wine_prediction_monitor")
   hopsworks_image = modal.Image.debian_slim().pip_install(["gradio", "requests", "hopsworks"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    dataset_api = project.get_dataset_api()

    dataset_api.download("Resources/images/df_recent.png", overwrite=True)
    dataset_api.download("Resources/images/confusion_matrix.png", overwrite=True)

    with gr.Blocks() as demo:     
        with gr.Row():
            with gr.Column():
                gr.Label("Recent Prediction History")
                input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
            with gr.Column():          
                gr.Label("Confusion Maxtrix with Historical Prediction Performance")
                input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")        

    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        with stub.run():
            f.remote()