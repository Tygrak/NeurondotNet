using System;
using System.Collections.Generic;

namespace NN{
    public class Layer : List<Neuron>{
        public Layer(int size){
            for (int i = 0; i < size; i++)
                base.Add(new Neuron());
        }
        public Layer(int size, Layer layer, Random rnd){
            for (int i = 0; i < size; i++)
                base.Add(new Neuron(layer, rnd));
        }
    }

    public class Weight{
        public Neuron inputNeuron;
        public double value;

        public Weight(){ 
        }

        public Weight(Neuron input, double value){
            this.inputNeuron = input;
            this.value = value;
        }
    }
}