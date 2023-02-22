import ml_ripeness.calculate_ripeness as cr


from keras.models import load_model



ripeness_model = load_model('ripeness_model.h5')



def main():
    cr.process_and_predict('test.png', ripeness_model)

main()