package main

import (
	"flag"
	"fmt"
	"math/rand"
	"path"
	"sync"

	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/plotutil"
	"github.com/gonum/plot/vg"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
	"github.com/wenkesj/evolution/agent"
	"github.com/wenkesj/evolution/env"
	"github.com/wenkesj/evolution/noise"
	"github.com/wenkesj/evolution/opt"
	"github.com/wenkesj/evolution/policy"
	"github.com/wenkesj/evolution/util"
)

var (
	creator = anyvec64.CurrentCreator()
	net     = newNet()
)

// newNet creates a new CartPole-v0 net
func newNet() anynet.Net {
	return anynet.Net{
		anynet.NewFC(creator, 4, 4),
		anynet.Tanh,
		anynet.NewFC(creator, 4, 4),
		anynet.Tanh,
		anynet.NewFC(creator, 4, 1),
		anynet.Tanh,
	}
}

// setNetParams sets the params of the network
func setNetParams(net anynet.Net, params []anyvec.Vector) {
	parameters := net.Parameters()
	for i, param := range parameters {
		param.Vector.Set(params[i])
	}
}

func main() {
	var baseURL, environment, monPath string
	var renderAgent, renderFinal bool
	var numAgents, globalStepLimit, episodeLimit, testRuns int
	var globalSeed int64
	var noiseStdDeviation, l2Coefficient, stepSize,
	 	beta1, beta2, epsilon, cutoffEpoch float64

	flag.StringVar(&baseURL, "url", "http://localhost:5000", "openai/gym-http-api url")
	flag.StringVar(&environment, "env", "CartPole-v0", "openai/gym environment")
	flag.StringVar(&monPath, "outmonitor", "", "path to save openai/gym environment monitor")
	flag.BoolVar(&renderAgent, "renderagent", false, "render openai/gym environment for agents (not recommended)")
	flag.BoolVar(&renderFinal, "renderfinal", false, "render openai/gym environment final test (recommended)")
	flag.IntVar(&numAgents, "agents", 2, "number of agents")
	flag.Int64Var(&globalSeed, "seed", 0, "random seed")
	flag.IntVar(&globalStepLimit, "steplimit", 100000, "openai/gym environment step limit")
	flag.IntVar(&episodeLimit, "episodes", 100, "number of episodes to run")
	flag.IntVar(&testRuns, "finalepisodes", 5, "number of episodes to run after training")
	flag.Float64Var(&cutoffEpoch, "cutoff", 180.0, "average agent cutoff training")
	flag.Float64Var(&noiseStdDeviation, "std", 0.02, "noise standard deviation")
	flag.Float64Var(&l2Coefficient, "l2", 0.005, "l2 regularization coefficient")
	flag.Float64Var(&stepSize, "stepsize", 0.01, "optimizer stepsize")
	flag.Float64Var(&beta1, "beta1", 0.9, "optimizer beta1 (adam)")
	flag.Float64Var(&beta2, "beta2", 0.999, "optimizer beta2 (adam)")
	flag.Float64Var(&epsilon, "epsilon", 1e-8, "optimizer epsilon (adam)")
	flag.Parse()

	// Global seeder for initialization
	seeder := rand.New(rand.NewSource(globalSeed))

	// Create the global policy
	p := policy.New(globalStepLimit)

	// Get the initial parameters of the network
	var paramsDimensions int
	parameters := net.Parameters()
	params := make([]anyvec.Vector, len(parameters))
	for i, param := range parameters {
		params[i] = param.Vector
		paramsDimensions += params[i].Len()
	}

	// Create the optimizer
	optimizer := opt.NewAdam(params, stepSize, beta1, beta2, epsilon)

	// Create the noise table
	noiseTable := noise.New(seeder.Int63(), numAgents*paramsDimensions)

	// Create agents
	agents := make([]*agent.Agent, numAgents)
	for i := range agents {
		client, id, err := env.New(baseURL, environment)
		if err != nil {
			panic(err)
		}

		agents[i] = agent.New(
			client, id, newNet(), rand.New(rand.NewSource(seeder.Int63())))
	}

	// Rollout episodes
	wg := new(sync.WaitGroup)
	var averageEpochs []float64
	for episode := 0; episode < episodeLimit; episode++ {
		fmt.Printf("\rEPISODE %d out of %d\n", episode, episodeLimit)
		fmt.Printf("\rAGENTS %d\n", len(agents))

		// Accumulate results
		var allRewards [][2]float64
		var allEpochs [][2]int

		// Share parameters to all agents and compute independently on buffered
		// channels
		rewards := make(chan [2]float64, len(agents))
		epochs := make(chan [2]int, len(agents))
		noiseVectors := make(chan anyvec.Vector, len(agents))

		for _, worker := range agents {
			wg.Add(1)
			go func(
				wg *sync.WaitGroup, worker *agent.Agent, params []anyvec.Vector) {
				defer wg.Done()
				// Make random perturbations
				posParams := make([]anyvec.Vector, len(params))
				negParams := make([]anyvec.Vector, len(params))
				noiseIndex := noiseTable.SampleIndex(worker.R, paramsDimensions)
				noiseVector := noiseTable.Chunk(noiseIndex, paramsDimensions)
				noiseVector.Scale(anyvec6