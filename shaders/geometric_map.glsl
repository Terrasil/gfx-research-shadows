// Project GSM: geometric mean of positive shifted depths, followed by a
// power-ratio response. Filter log depth, not blocker gaps or PCF results.
float geometricMapEncode(float depth){return log(1.0+clamp(depth,0.0,1.0));}
float geometricMapVisibility(float meanLogDepth,float receiverDepth,float sharpness){
    float receiverLog=log(1.0+max(receiverDepth,0.0));
    return exp(min(0.0,sharpness*(meanLogDepth-receiverLog)));
}
