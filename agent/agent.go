// Package agent implements a neural network worker.
package agent

import (
	gym "github.com/openai/gym-http-api/binding-go"
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec/anyvec64"
	"math/rand"
)

// Agent is a container for an environment/instance and works independently.
type Agent struct {
	Client *gym.Client
	Id     gym.InstanceID
	R      *rand.Rand
	Net    anynet.Net
}

// New creates a new Agent
func New(
	client *gym.Client, id gym.InstanceID, net anynet.Net, r *rand.Rand) *Agent {
	agent := new(Agent)
	agent.Client = client
	agent.Id = id
	agent.R = r
	agent.Net = net
	return agent
}

// Action returns an action from the given observation
func (agent *Agent) Action(observation interface{}) interface{} {
	// Set the parameters of the network from the list of par