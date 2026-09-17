package com.tencent.yoloncnn;

public class Object {
    private int label;
    private float prob;

    public Object(int label, float prob) {
        this.label = label;
        this.prob = prob;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public float getProb() {
        return prob;
    }

    public void setProb(float prob) {
        this.prob = prob;
    }

    @Override
    public String toString() {
        return "Object{" +
                "label=" + label +
                ", prob=" + prob +
                '}';
    }
}
