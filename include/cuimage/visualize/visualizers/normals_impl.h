
/**
 * Specialization for NORMALS_TYPE
 */
template <>
class TypedVisualizer<NORMALS_TYPE> : public VisualizerBase
{
public:
    TypedVisualizer(const std::string& name, const size_t w, const size_t h)
     : VisualizerBase(name, w, h)
    {
    }
private:
    virtual bool initTexture_() override
    {
        if (texture_.IsValid())
            return true;
        
        texture_.Reinitialise(w_, h_, GL_RGB32F, true, 0, GL_RGB, GL_FLOAT);
        return texture_.IsValid();
    }

    virtual bool copyToTexture_() override
    {
        assert(uploadDataSize_ == 3 * w_ * h_ * sizeof(float));
        
        // Copy the data to not modifiy original data
        float3* tmp;
        cudaSafeCall(cudaMalloc(&tmp, uploadDataSize_));
        cudaSafeCall(cudaMemcpy(tmp, uploadData_, uploadDataSize_, cudaMemcpyDeviceToDevice));

        // Modify the data to be [0 - 1] -> (data + 1) / 2
        DevPtr<float3> data(tmp, w_, h_);
        cu_AddTo(data, make_float3(1.f, 1.f, 1.f));
        cu_DivideBy(data, make_float3(2.f, 2.f, 2.f));

        return copyToArray<float3, float4>(data.data, array_, w_, h_);
    }   
};