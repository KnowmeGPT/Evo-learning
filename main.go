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
