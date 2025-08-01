package models

type Problem struct {
	Id         string `json:"id"`
	Uid        string `json:"uid"`
	TargetQps  int    `json:"targetqps"`
	MaxQps     int    `json:"-"`
	EntryPoint string `json:"entrypoint"`
}