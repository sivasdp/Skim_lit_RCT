import tensorflow as tf

def loaded_model(x,tf):
    loaded_trihybrid_model = tf.keras.models.load_model("skimlit_tribrid_model")
    loaded_model_pred_probs = loaded_trihybrid_model.predict(x)
    loaded_model_preds = tf.argmax(loaded_model_pred_probs, axis=1)
    class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    class_val = []
    for val in loaded_model_preds:
        class_val.append(class_names[val])
    return class_val
