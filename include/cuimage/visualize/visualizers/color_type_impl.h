
/**
 * Specialization for COLOR_TYPE_GREY
 */
template <>
class TypedVisualizer<COLOR_TYPE_GREY> : public VisualizerBase
{
public:
    TypedVisualizer(const std::string& name, const size_t w, const size_t h)
     : VisualizerBase(name, w, h)
    {
    }

    virtual bool initTexture_() override
    {
        if (texture_.IsValid())
            return true;

        assert(uploadDataSize_ == w_ * h_ * sizeof(unsigned char));
        texture_.Reinitialise(w_, h_, GL_LUMINANCE8, true, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE);
        return texture_.IsValid();
    }
};

/**
 * Specialization for COLOR_TYPE_RGB
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGB> : public VisualizerBase
{
public:
    TypedVisualizer(const std::string& name, const size_t w, const size_t h)
     : VisualizerBase(name, w, h)
    {
    }

protected:
    virtual bool initTexture_() override
    {
        pangolin::BindToContext(name_);
        
        if (texture_.IsValid())
            return true;
            
        assert(uploadDataSize_ == 3 * w_ * h_ * sizeof(unsigned char));
        texture_.Reinitialise(w_, h_, GL_RGB8, true, 0, GL_RGB, GL_UNSIGNED_BYTE);
        return texture_.IsValid();
    }

    virtual bool copyToTexture_() override
    {
        // Texture will be RGBA, so we need to inflate
        assert(uploadDataSize_ == 3 * w_ * h_ * sizeof(unsigned char));
        return copyToArray<uchar3, uchar4>((uchar3*)uploadData_, array_, w_, h_);
    }
};

/**
 * Specialization for COLOR_TYPE_RGBA
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGBA> : public VisualizerBase
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

        assert(uploadDataSize_ == 4 * w_ * h_ * sizeof(unsigned char));
        texture_.Reinitialise(w_, h_, GL_RGBA, true, 0, GL_RGBA, GL_UNSIGNED_BYTE);
        return texture_.IsValid();
    }
};

/**
 * Specialization for COLOR_TYPE_GREY_F
 */
template <>
class TypedVisualizer<COLOR_TYPE_GREY_F> : public VisualizerBase
{
public:
    TypedVisualizer(const std::string& name, const size_t w, const size_t h)
     : VisualizerBase(name, w, h)
    {
    }

    virtual bool initTexture_() override
    {
        if (texture_.IsValid())
            return true;
        
        assert(uploadDataSize_ == w_ * h_ * sizeof(float));
        texture_.Reinitialise(w_, h_, GL_LUMINANCE32F_ARB, true, 0, GL_LUMINANCE, GL_FLOAT);
        return texture_.IsValid();
    }
};

/**
 * Specialization for COLOR_TYPE_RGB_F
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGB_F> : public VisualizerBase
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
        // Texture will be RGBA, so we need to inflate
        assert(uploadDataSize_ == 3 * w_ * h_ * sizeof(float));
        return copyToArray<float3, float4>((float3*)uploadData_, array_, w_, h_);
    }   
};

/**
 * Specialization for COLOR_TYPE_RGBA_F
 */
template <>
class TypedVisualizer<COLOR_TYPE_RGBA_F> : public VisualizerBase
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
        
        assert(uploadDataSize_ == 4 * w_ * h_ * sizeof(float));
        texture_.Reinitialise(w_, h_, GL_RGBA32F, true, 0, GL_RGBA, GL_FLOAT);
        return texture_.IsValid();
    } 
};