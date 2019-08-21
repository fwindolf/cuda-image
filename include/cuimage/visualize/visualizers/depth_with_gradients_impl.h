/**
 * Specialization for DEPTH_WITH_GRADIENTS_TYPE
 */

template <>
class TypedVisualizer<DEPTH_WITH_GRADIENTS_TYPE> : public VisualizerBase
{
public:
    TypedVisualizer(const std::string& name, const size_t w, const size_t h);

    virtual bool initContext_() override;

    virtual bool bindToTexture_() override;

    virtual bool initTexture_() override;

    virtual bool copyToTexture_() override;

    virtual void render_() override;

private:
    float factor_ = (1280 / w_);
    float fx_ = 1320.f / factor_;
    float fy_ = 1320.f / factor_;
    float cx_ = w_ / 2.f;
    float cy_ = h_ / 2.f;

    pangolin::OpenGlRenderState s_cam_;
    pangolin::View d_cam_;

    pangolin::GlBufferCudaPtr vertices_;
    pangolin::GlBufferCudaPtr normals_;
    pangolin::GlBufferCudaPtr indices_;

    pangolin::GlSlProgram shader_;
};

/**
 * DEPTH_WITH_GRADIENTS_TYPE
 */

inline TypedVisualizer<DEPTH_WITH_GRADIENTS_TYPE>::TypedVisualizer(
    const std::string& name, const size_t w, const size_t h)
    : VisualizerBase(name, w, h)
{
}

inline bool TypedVisualizer<DEPTH_WITH_GRADIENTS_TYPE>::initContext_()
{
    // Create 3D view
    s_cam_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(w_, h_, fx_, fy_, cx_, cy_, 0.1, 5000),
        pangolin::ModelViewLookAt(0, 0, 0, 0, 0, 10, pangolin::AxisNegY)
        // pangolin::ModelViewLookAt(1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        // pangolin::AxisY)
    );

    d_cam_ = pangolin::Display("cam")
                 .SetBounds(0, 1.0f, 0, 1.0f, -(float)w_ / h_)
                 .SetHandler(new pangolin::Handler3D(s_cam_));

    // Enable custom shader
    shader_.ClearShaders();
    shader_.AddShader(
        pangolin::GlSlAnnotatedShader, pangolin::ambient_light_shader);
    // shader_.AddShader(pangolin::GlSlAnnotatedShader,
    // pangolin::default_model_shader);
    bool success = shader_.Link();
    if (!success)
        std::cerr << "Ambient Light shader was not compiled correctly or did "
                     "not link!"
                  << std::endl;

    return success;
}

inline bool TypedVisualizer<DEPTH_WITH_GRADIENTS_TYPE>::initTexture_()
{
    if (vertices_.IsValid() && indices_.IsValid())
        return true;

    assert(uploadDataSize_ == 3 * w_ * h_ * sizeof(float));

    vertices_.Reinitialise(pangolin::GlBufferType::GlArrayBuffer, 2 * w_ * h_,
        GL_FLOAT, 4, cudaGraphicsMapFlagsWriteDiscard, GL_STATIC_DRAW);

    indices_.Reinitialise(pangolin::GlBufferType::GlElementArrayBuffer,
        6 * w_ * h_, GL_UNSIGNED_INT, 1, cudaGraphicsMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    return (vertices_.IsValid() && indices_.IsValid());
}

inline bool TypedVisualizer<DEPTH_WITH_GRADIENTS_TYPE>::bindToTexture_()
{
    return true;
}

inline bool TypedVisualizer<DEPTH_WITH_GRADIENTS_TYPE>::copyToTexture_()
{
    pangolin::CudaScopedMappedPtr vertex_array(vertices_);
    DevPtr<float3> data((float3*)uploadData_, w_, h_);
    DevPtr<float4> verts((float4*)*vertex_array, w_, 2 * h_);
    cu_VerticesFromDepthGradients(verts, data, fx_, fy_, cx_, cy_);

    // Generate the indices of connected triangles to the right[0 1 0+w], and
    // to the left [1 1+w 1+w-1] to connect all points
    pangolin::CudaScopedMappedPtr index_array(indices_);
    DevPtr<unsigned int> inds((unsigned int*)*index_array, w_, 6 * h_);
    cu_VertexIndices<float3>(inds, data);

    return true;
}

inline void TypedVisualizer<DEPTH_WITH_GRADIENTS_TYPE>::render_()
{
    // Render triangles for every
    d_cam_.Activate(s_cam_);

    glClearColor(0.1f, 0.1f, 0.1f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    if (shader_.Valid())
    {
        shader_.Bind();

        // Uniforms for vertex shader
        Eigen::Matrix4f model
            = Eigen::Matrix4f::Identity(); // No model transformation
        Eigen::Matrix4f view = s_cam_.GetModelViewMatrix();
        Eigen::Matrix4f projection = s_cam_.GetProjectionMatrix();
        shader_.SetUniform("model", model);
        shader_.SetUniform("view", view);
        shader_.SetUniform("projection", projection);

        // Uniform for fragment shader
        shader_.SetUniform("objectColor", 1.0f, 1.0f, 1.0f);
        shader_.SetUniform("lightColor", 1.f, 1.f, 1.f);
        shader_.SetUniform("lightPos", 0.f, 0.f, 0.f);
        shader_.SetUniform("ambient", .3f, .3f, .3f);

        vertices_.Bind();
        size_t stride = vertices_.count_per_element
            * pangolin::GlDataTypeBytes(vertices_.datatype);

        // Vertices: 0, 1, 2, 3,   8, 9, 10, 11 ...
        glVertexAttribPointer(0, vertices_.count_per_element,
            vertices_.datatype, GL_FALSE, 2 * stride, (void*)0);
        glEnableVertexAttribArray(0);

        // Normals: 4, 5, 6, 7,   12, 13, 14, 15 ...
        glVertexAttribPointer(1, vertices_.count_per_element,
            vertices_.datatype, GL_FALSE, 2 * stride, (void*)stride);
        glEnableVertexAttribArray(1);

        indices_.Bind();
        glDrawElements(
            GL_TRIANGLES, indices_.num_elements, indices_.datatype, 0);
        indices_.Unbind();

        glDisableClientState(GL_VERTEX_ARRAY);
        normals_.Unbind();
        vertices_.Unbind();

        shader_.Unbind();
    }

    glDisable(GL_CULL_FACE);
}
