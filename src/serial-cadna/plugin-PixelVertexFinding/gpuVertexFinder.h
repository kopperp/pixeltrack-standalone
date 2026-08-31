#ifndef RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h
#define RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h

#include <cadna.h>

#include <cstddef>
#include <cstdint>

#include "CUDADataFormats/ZVertexHeterogeneous.h"

namespace gpuVertexFinder {

  using ZVertices = ZVertexSoA;
  using TkSoA = pixelTrack::TrackSoA;

  // workspace used in the vertex reco algos
  struct WorkSpace {
    static constexpr uint32_t MAXTRACKS = ZVertexSoA::MAXTRACKS;
    static constexpr uint32_t MAXVTX = ZVertexSoA::MAXVTX;

    uint32_t ntrks;            // number of "selected tracks"
    uint16_t itrk[MAXTRACKS];  // index of original track
    float_st zt[MAXTRACKS];    // input track z at bs
    float_st ezt2[MAXTRACKS];  // input error^2 on the above
    float_st ptt2[MAXTRACKS];  // input pt^2 on the above
    uint8_t izt[MAXTRACKS];    // interized z-position of input tracks
    int32_t iv[MAXTRACKS];     // vertex index for each associated track

    uint32_t nvIntermediate;  // the number of vertices after splitting pruning etc.

    void init() {
      ntrks = 0;
      nvIntermediate = 0;
    }
  };

  inline void init(ZVertexSoA* pdata, WorkSpace* pws) {
    pdata->init();
    pws->init();
  }

  class Producer {
  public:
    using ZVertices = ZVertexSoA;
    using WorkSpace = gpuVertexFinder::WorkSpace;
    using TkSoA = pixelTrack::TrackSoA;

    Producer(bool oneKernel,
             bool useDensity,
             bool useDBSCAN,
             bool useIterative,
             int iminT,         // min number of neighbours to be "core"
             float_st ieps,     // max absolute distance to cluster
             float_st ierrmax,  // max error to be "seed"
             float_st ichi2max  // max normalized distance to cluster
             )
        : oneKernel_(oneKernel && !(useDBSCAN || useIterative)),
          useDensity_(useDensity),
          useDBSCAN_(useDBSCAN),
          useIterative_(useIterative),
          minT(iminT),
          eps(ieps),
          errmax(ierrmax),
          chi2max(ichi2max) {}

    ~Producer() = default;

    ZVertexHeterogeneous makeAsync(cudaStream_t stream, TkSoA const* tksoa, float_st ptMin) const;
    ZVertexHeterogeneous make(TkSoA const* tksoa, float_st ptMin) const;

  private:
    const bool oneKernel_;
    const bool useDensity_;
    const bool useDBSCAN_;
    const bool useIterative_;

    int minT;          // min number of neighbours to be "core"
    float_st eps;      // max absolute distance to cluster
    float_st errmax;   // max error to be "seed"
    float_st chi2max;  // max normalized distance to cluster
  };

}  // namespace gpuVertexFinder

#endif  // RecoPixelVertexing_PixelVertexFinding_src_gpuVertexFinder_h
