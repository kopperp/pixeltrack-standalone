//
// Author: Felice Pantaleo, CERN
//

// #define BROKENLINE_DEBUG

#include <cstdint>

#include <Framework/Cadna.h>

#include "CUDACore/cudaCompat.h"

#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/cuda_assert.h"
#include "CondFormats/pixelCPEforGPU.h"

#include "BrokenLine.h"
#include "HelixFitOnGPU.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using Tuples = pixelTrack::HitContainer;
using OutputSoA = pixelTrack::TrackSoA;

// #define BL_DUMP_HITS

template <int N>
void kernelBLFastFit(Tuples const *__restrict__ foundNtuplets,
                     CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                     HitsOnGPU const *__restrict__ hhp,
                     double_st *__restrict__ phits,
                     float_st *__restrict__ phits_ge,
                     double_st *__restrict__ pfast_fit,
                     uint32_t nHits,
                     uint32_t offset) {
  constexpr uint32_t hitsInFit = N;

  assert(hitsInFit <= nHits);

  assert(hhp);
  assert(pfast_fit);
  assert(foundNtuplets);
  assert(tupleMultiplicity);

  // look in bin for this hit multiplicity
  auto local_start = 0;

#ifdef BROKENLINE_DEBUG
  if (0 == local_start) {
    printf("%d total Ntuple\n", foundNtuplets->nbins());
    printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
  }
#endif

  for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt; local_idx++) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it from the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);
    assert(tkid < foundNtuplets->nbins());

    assert(foundNtuplets->size(tkid) == nHits);

    Rfit::Map3xNd<N> hits(phits + local_idx);
    Rfit::Map4d fast_fit(pfast_fit + local_idx);
    Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

#ifdef BL_DUMP_HITS
    int done;
    done = 0;

    bool dump = (foundNtuplets->size(tkid) == 5 && 0 == atomicAdd(&done, 1));
#endif

    // Prepare data structure
    auto const *hitId = foundNtuplets->begin(tkid);
    for (unsigned int i = 0; i < hitsInFit; ++i) {
      auto hit = hitId[i];
      float_st ge[6];

#if CADNA_DEBUG
      const int in_digits = ge->nb_significant_digit();
#endif

      hhp->cpeParams()
          .detParams(hhp->detectorIndex(hit))
          .frame.toGlobal(hhp->xerrLocal(hit), 0, hhp->yerrLocal(hit), ge);
#ifdef BL_DUMP_HITS
      if (dump) {
        printf("Hit global: %d: %d hits.col(%d) << %f,%f,%f\n",
               tkid,
               hhp->detectorIndex(hit),
               i,
               hhp->xGlobal(hit),
               hhp->yGlobal(hit),
               hhp->zGlobal(hit));
        printf("Error: %d: %d  hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n",
               tkid,
               hhp->detetectorIndex(hit),
               i,
               ge[0],
               ge[1],
               ge[2],
               ge[3],
               ge[4],
               ge[5]);
      }
#endif
      hits.col(i) << hhp->xGlobal(hit), hhp->yGlobal(hit), hhp->zGlobal(hit);
      hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];

#if CADNA_DEBUG
      std::cout << "-- BROKEN LINE FIT COORDINATES (FAST) --" << std::endl;
      print_cadna_metric("Coordinate (x)", hhp->xGlobal(hit), -1);
      print_cadna_metric("Coordinate (y)", hhp->yGlobal(hit), -1);
      print_cadna_metric("Coordinate (z)", hhp->zGlobal(hit), -1);

      for (int i = 0; i<6; ++i)
        print_cadna_metric(std::format("Covariance ({})", i+1), ge[i], in_digits);
#endif
    }
    BrokenLine::BL_Fast_fit(hits, fast_fit);

#if CADNA_DEBUG
    std::cout << "-- BROKEN LINE FIT (FAST) --" << std::endl;
    print_cadna_metric("Center offset (x)", fast_fit.coeff(0), -1);
    print_cadna_metric("Center offset (y)", fast_fit.coeff(1), -1);
    print_cadna_metric("Radius", fast_fit.coeff(2), -1);
    print_cadna_metric("Track slope", fast_fit.coeff(3), -1);
#endif

    // no NaN here....
    assert(fast_fit(0) == fast_fit(0));
    assert(fast_fit(1) == fast_fit(1));
    assert(fast_fit(2) == fast_fit(2));
    // assert(fast_fit(3) == fast_fit(3));
  }
}

template <int N>
void kernelBLFit(CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                 double_st B,
                 OutputSoA *results,
                 double_st *__restrict__ phits,
                 float_st *__restrict__ phits_ge,
                 double_st *__restrict__ pfast_fit,
                 uint32_t nHits,
                 uint32_t offset) {
  assert(N <= nHits);

  assert(results);
  assert(pfast_fit);

  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = 0;
  for (int local_idx = local_start, nt = Rfit::maxNumberOfConcurrentFits(); local_idx < nt; local_idx++) {
    auto tuple_idx = local_idx + offset;
    if (tuple_idx >= tupleMultiplicity->size(nHits))
      break;

    // get it for the ntuple container (one to one to helix)
    auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_idx);

    Rfit::Map3xNd<N> hits(phits + local_idx);
    Rfit::Map4d fast_fit(pfast_fit + local_idx);
    Rfit::Map6xNf<N> hits_ge(phits_ge + local_idx);

    BrokenLine::PreparedBrokenLineData<N> data;
    Rfit::Matrix3d Jacob;

    BrokenLine::karimaki_circle_fit circle;
    Rfit::line_fit line;

#if CADNA_DEBUG
    const int in_digits = hits.col(0)(0).nb_significant_digit();
#endif

    BrokenLine::prepareBrokenLineData(hits, fast_fit, B, data);
    BrokenLine::BL_Line_fit(hits_ge, fast_fit, B, data, line);
    BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle);

    results->stateAtBS.copyFromCircle(circle.par, circle.cov, line.par, line.cov, 1.f / float(B), tkid);
    results->pt(tkid) = float(B) / float(abs(circle.par(2)));
    results->eta(tkid) = asinhf(line.par(0));
    results->chi2(tkid) = (circle.chi2 + line.chi2) / (2 * N - 5);

#if CADNA_DEBUG
    {
      Eigen::Matrix<double, 3, N> hits_raw = Rfit::Map3xNd<N>(phits + local_idx).template cast<double>();
      Eigen::Vector4d fast_fit_raw = Rfit::Map4d(pfast_fit + local_idx).template cast<double>();
      Eigen::Matrix<float, 6, N> hits_ge_raw = Rfit::Map6xNf<N>(phits_ge + local_idx).template cast<float>();

      Eigen::Matrix<double_st, 3, N> hits_dbl = hits_raw.template cast<double_st>();
      Eigen::Vector<double_st, 4> fast_fit_dbl = fast_fit_raw.template cast<double_st>();
      Eigen::Matrix<float_st, 6, N> hits_ge_dbl = hits_ge_raw.template cast<float_st>();

      const int in_digits_dbl = hits_dbl.coeff(0).nb_significant_digit();

      BrokenLine::PreparedBrokenLineData<N> data_dbl;

      BrokenLine::karimaki_circle_fit circle_dbl;
      Rfit::line_fit line_dbl;

      BrokenLine::prepareBrokenLineData(hits_dbl, fast_fit_dbl, static_cast<double>(B), data_dbl);
      BrokenLine::BL_Line_fit(hits_ge_dbl, fast_fit_dbl, B, data_dbl, line_dbl);
      BrokenLine::BL_Circle_fit(hits_dbl, hits_ge_dbl, fast_fit_dbl, B, data_dbl, circle_dbl);

      std::cout << "-- BROKEN LINE FIT --\n";
      print_cadna_metric("Circle Phi", circle.par(0), -1, circle_dbl.par(0), in_digits_dbl);
      print_cadna_metric("Circle d_ca", circle.par(1), -1, circle_dbl.par(1), in_digits_dbl);
      print_cadna_metric("Circle Curvature k", circle.par(2), in_digits, circle_dbl.par(2), in_digits_dbl);
      print_cadna_metric("Circle Chi2", circle.chi2, -1, circle_dbl.chi2, in_digits_dbl);
      print_cadna_metric("Line Chi2", line.chi2, -1, line_dbl.chi2, in_digits_dbl);
      print_cadna_metric("Pt", results->pt(tkid), in_digits);
      print_cadna_metric("Eta", results->eta(tkid), in_digits);
    }
#endif

#ifdef BROKENLINE_DEBUG
    if (!(circle.chi2 >= 0) || !(line.chi2 >= 0))
      printf("kernelBLFit failed! %f/%f\n", circle.chi2, line.chi2);
    printf("kernelBLFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
           N,
           nHits,
           tkid,
           circle.par(0),
           circle.par(1),
           circle.par(2));
    printf("kernelBLHits line.par(0,1): %d %f,%f\n", tkid, line.par(0), line.par(1));
    printf("kernelBLHits chi2 cov %f/%f  %e,%e,%e,%e,%e\n",
           circle.chi2,
           line.chi2,
           circle.cov(0, 0),
           circle.cov(1, 1),
           circle.cov(2, 2),
           line.cov(0, 0),
           line.cov(1, 1));
#endif
  }
}
