# Evo-learning
This is a local, non-distributed, Go implementation of the [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864). The original work from the paper can be found at [openai/evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter).

It uses the [openai/gym-http-api](https://github.com/openai/gym-http-api), [binding-go](https://github.com/openai/gym-http-api/tree/master/binding-go), and [unixpickle/anynet](https://github.com/unixpickle/anynet) and [unixpickle/anyvec](https://github.com/unixpickle/anyvec) for efficient high-level vector computation.

## Instructions
The project is aimed to solve [CartPole-v0](https://gym.openai.com/envs/CartPole-v0), which requires 195 epochs/reward over 100 episodes.

First, clone or download the repo:
```sh
$ go get github.com/KnowmeGPT/evolution
```
In a separate terminal, open gym from the `github.com/openai/gym-http-api` directory in your file system.
```
$ python gym_http_server.py
```
Now, run the trainer and evaluator with the specifications of your choice.

```sh
$ # 200 episodes of training by 2 agents and 100 episodes of evaluation with a single agent
$ # Saving results to a directory "~/agents2eps200"
$ evolution --outmonitor ~/agents2eps200 --finalepisodes 100 --episodes 200 --agents 2
```
## Results
After a trial and error phase of 42 episodes, the agents evolve enough to dominate the game independently. We apply a cutoff average reward of 195 or more for both agents, implying that the parameters should typically be capable of solving the game independently. The average reward achieved is 198.5 over 100 episodes!

## Future Plans
- Parallelize where needed
- Add 32/64 bit support
- Support multiple environments
- Allow network serialization/deserialization
- Optimize code further
- Include plotting, statistics, performance profiling, and uploading capabilities

## Disclaimer
This is a project initially created for a [Complex Systems and Networks](http://www.ece.uc.edu/~aminai/EECE7065.pdf) class. It is not intended to compare with the original work. Rather, it is an execution with results and interpretation.

