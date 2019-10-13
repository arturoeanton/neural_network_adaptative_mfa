package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	deep "github.com/patrikeh/go-deep"
	training "github.com/patrikeh/go-deep/training"
)

var (
	neural *deep.Neural
	base   string
)

func main() {
	rand.Seed(time.Now().UnixNano())
	base = "http://127.0.0.1:9090"
	/*
		Input esta definido por
		Distancia de la ultima ip en km
		Tiempo de la ultima coneccion en minutos
		Riesgo de la ip actual de 1 a 0
		Riesgo de la anterior ip de 1 a 0
	*/

	data := getData("data")

	fmt.Println("Load data ok")
	for i := range data {
		deep.Standardize(data[i].Input)
	}
	data.Shuffle()
	fmt.Println("Standardize ok")
	neural = deep.NewNeural(&deep.Config{
		/* Input dimensionality */
		Inputs: len(data[0].Input),
		/* Two hidden layers consisting of two neurons each, and a single output */
		Layout: []int{8, 8, 6, 6, 4, 3, 1},
		/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
		Activation: deep.ActivationSigmoid,
		/* Determines output layer activation & loss function:
		ModeRegression: linear outputs with MSE loss
		ModeMultiClass: softmax output with Cross Entropy loss
		ModeMultiLabel: sigmoid output with Cross Entropy loss
		ModeBinary: sigmoid output with binary CE loss */
		Mode: deep.ModeMultiLabel,
		/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
		Weight: deep.NewNormal(1.0, 0.0),
		/* Apply bias */
		Bias: false,
	})

	// params: learning rate, momentum, alpha decay, nesterov

	//optimizer := training.NewSGD(0.8, 0.1, 0.001, true)
	//optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
	//optimizer := training.NewAdam(0.7, 0.9, 0.999, 1e-8)

	//optimizer := training.NewSGD(0.07, 0.1, 1e-6, true)

	// params: optimizer, verbosity (print stats at every 50th iteration)

	//optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
	optimizer := training.NewSGD(0.07, 0.0, 0.0, false)
	// params: optimizer, verbosity (print info at every n:th iteration), batch-size, number of workers
	// trainer := training.NewBatchTrainer(optimizer, 1, 200, 50)
	trainer := training.NewTrainer(optimizer, 100)

	training, heldout := data.Split(0.75)
	fmt.Println("len training:", len(training))
	fmt.Println("len heldout:", len(heldout))
	trainer.Train(neural, training, heldout, 2000)

	fmt.Println("Data: data.json")
	printError("data")
	fmt.Println("Data: data1.json")
	printError("data1")
	fmt.Println("Data: data2.json")
	printError("data2")

	r := mux.NewRouter()
	r.HandleFunc("/predict", predict)
	r.HandleFunc("/data/{name}", show)

	http.Handle("/", r)
	log.Fatal(http.ListenAndServe(":9090", nil))

}

func show(w http.ResponseWriter, r *http.Request) {
	name := mux.Vars(r)["name"]
	data := getData(name)
	w.Header().Set("Content-Type", "application/json")
	ret := []string{}
	for _, e := range data {
		ret = append(ret, base+"/predict?input="+strings.Trim(strings.Join(strings.Fields(fmt.Sprint(e.Input)), ","), "[]"))
	}

	json.NewEncoder(w).Encode(ret)
}

func predict(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	var err error
	inputstr := strings.Split(r.URL.Query().Get("input"), ",")
	input := []float64{0, 0, 0, 0}
	for i, e := range inputstr {
		input[i], err = strconv.ParseFloat(e, 64)
		if err != nil {
			json.NewEncoder(w).Encode(err)
			return
		}
	}
	fmt.Println(input)
	deep.Standardize(input)
	json.NewEncoder(w).Encode(neural.Predict(input))
}

func generateData(name string) {
	var data training.Examples
	for i := 0; i < 2500; i++ {
		km := rand.Intn(30000-2) + 2
		min := rand.Intn(30000-1) + 1
		risk1 := rand.Float64()
		risk2 := rand.Float64()
		data = append(data, calculateData(float64(km), float64(min), risk1, risk2))
	}
	for i := 0; i < 2500; i++ {
		km := rand.Intn(30000-2) + 2
		min := rand.Intn(30000-1) + 1
		risk1 := 0.0
		risk2 := 0.0
		data = append(data, calculateData(float64(km), float64(min), risk1, risk2))
	}
	for i := 0; i < 2500; i++ {
		km := rand.Intn(30000-2) + 2
		min := rand.Intn((1500)-1) + 1
		risk1 := 0.0
		risk2 := 0.0
		data = append(data, calculateData(float64(km), float64(min), risk1, risk2))
	}
	for i := 0; i < 2500; i++ {
		km := rand.Intn(100-2) + 2
		min := rand.Intn((1500)-1) + 1
		risk1 := 0.0
		risk2 := 0.0
		data = append(data, calculateData(float64(km), float64(min), risk1, risk2))
	}

	file, _ := json.MarshalIndent(data, "", " ")
	_ = ioutil.WriteFile(name+".json", file, 0644)
	fmt.Println("save data in ./" + name + ".json")
}

func calculateData(km float64, min float64, risk1 float64, risk2 float64) training.Example {
	var out float64
	out = 0
	if risk1 > 0.5 {
		out = 1.0
	}
	if risk2 > 0.5 {
		out = 1.0
	}

	if risk2 > 0.5 {
		out = 1.0
	}

	if float64(min/km) < 0.6 {
		out = 1.0
	}
	return training.Example{
		Input:    []float64{float64(km), float64(min), risk1, risk2},
		Response: []float64{out},
	}

}

func getData(name string) training.Examples {
	var data training.Examples
	jsonFile, err := os.Open(name + ".json")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened " + name + ".json")
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	json.Unmarshal(byteValue, &data)
	return data
}

func printError(name string) {
	data := getData(name)
	errorN := 0
	error0 := 0
	error1 := 0
	for _, d := range data {
		deep.Standardize(d.Input)
		if math.Round(neural.Predict(d.Input)[0]) != d.Response[0] {
			//fmt.Println(d.Input, "\t\t->", math.Round(neural.Predict(d.Input)[0]), " real is ", d.Response[0])
			errorN++
			if math.Round(neural.Predict(d.Input)[0]) == 1 {
				error1++
			}
			if math.Round(neural.Predict(d.Input)[0]) == 0 {
				error0++
			}
		}
	}
	errorPorc := float64(errorN) / float64(len(data))
	fmt.Println("eP:", errorPorc, " - eN:", errorN, " - e0:", error0, " - e1:", error1, " - len data:", len(data))

}
