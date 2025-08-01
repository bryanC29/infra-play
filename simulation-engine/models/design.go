package models

type Design struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

type Node struct {
	Id     string     `json:"id"`
	Type   string     `json:"type"`
	Failed bool       `json:"-"`
	Config NodeConfig `json:"config"`
}

type Edge struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Source *Node  `json:"-"`
	Target *Node  `json:"-"`
}

type NodeConfig struct {
	VCpu    int `json:"vcpu"`
	Mem     int `json:"mem"`
	Replica int `json:"replica"`
}

type Graph struct {
	Nodes map[string]*Node
	Edges map[string][]*Node
}