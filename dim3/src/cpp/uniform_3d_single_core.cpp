#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;


void update_EtoV(
        int nx, int ny, int nz,
        py::array_t<int> &d_ijk2nodeTags,
        py::array_t<int> &d_EtoV) {
    
    auto ijk2nodeTags = d_ijk2nodeTags.unchecked<3>();
    auto EtoV = d_EtoV.mutable_unchecked<2>();
    
    int node0, node1, node2, node3, node4, node5, node6, node7;
    int elemTag = 0;
    
    for (int i=0; i<nx-1; i++) {
        for (int j=0; j<ny-1; j++) {
            for (int k=0; k<nz-1; k++) {
                // nodeTags of a hexahedron cube
                node0 = ijk2nodeTags(i  ,j  ,k  );
                node1 = ijk2nodeTags(i  ,j  ,k+1);
                node2 = ijk2nodeTags(i  ,j+1,k  );
                node3 = ijk2nodeTags(i  ,j+1,k+1);
                node4 = ijk2nodeTags(i+1,j  ,k  );
                node5 = ijk2nodeTags(i+1,j  ,k+1);
                node6 = ijk2nodeTags(i+1,j+1,k  );
                node7 = ijk2nodeTags(i+1,j+1,k+1);

                // EtoV: nodeTags(vertices) of 6 tetrahedrons in a hexahedral cube
                EtoV(elemTag  ,0) = node0;
                EtoV(elemTag  ,1) = node7;
                EtoV(elemTag  ,2) = node3;
                EtoV(elemTag  ,3) = node1;
                
                EtoV(elemTag+1,0) = node0;
                EtoV(elemTag+1,1) = node7;
                EtoV(elemTag+1,2) = node2;
                EtoV(elemTag+1,3) = node3;
                
                EtoV(elemTag+2,0) = node0;
                EtoV(elemTag+2,1) = node7;
                EtoV(elemTag+2,2) = node6;
                EtoV(elemTag+2,3) = node2;
                
                EtoV(elemTag+3,0) = node0;
                EtoV(elemTag+3,1) = node7;
                EtoV(elemTag+3,2) = node4;
                EtoV(elemTag+3,3) = node6;
                
                EtoV(elemTag+4,0) = node0;
                EtoV(elemTag+4,1) = node7;
                EtoV(elemTag+4,2) = node5;
                EtoV(elemTag+4,3) = node4;
                
                EtoV(elemTag+5,0) = node0;
                EtoV(elemTag+5,1) = node7;
                EtoV(elemTag+5,2) = node1;
                EtoV(elemTag+5,3) = node5;
                
                elemTag += 6;
            }
        }
    }
}


void update_EtoEFRB(
        int nx, int ny, int nz,
        py::array_t<int> &d_xidxs,
        py::array_t<int> &d_yidxs,
        py::array_t<int> &d_zidxs,
        py::array_t<int> &d_ijk2cubeTags,
        py::array_t<int> &d_phyTags,
        py::array_t<int> &d_EtoE,
        py::array_t<int> &d_EtoF,
        py::array_t<int> &d_EtoR,
        py::array_t<int> &d_EtoB) {
    
    auto xidxs = d_xidxs.unchecked<1>();
    auto yidxs = d_yidxs.unchecked<1>();
    auto zidxs = d_zidxs.unchecked<1>();
    auto ijk2cubeTags = d_ijk2cubeTags.unchecked<3>();
    auto phyTags = d_phyTags.unchecked<1>();
    auto EtoE = d_EtoE.mutable_unchecked<2>();
    auto EtoF = d_EtoF.mutable_unchecked<2>();
    auto EtoR = d_EtoR.mutable_unchecked<2>();
    auto EtoB = d_EtoB.mutable_unchecked<2>();
    
    int cubeTag;
    int cubeTag_xM, cubeTag_xP;
    int cubeTag_yM, cubeTag_yP;
    int cubeTag_zM, cubeTag_zP;
    int elemTag = 0;
    
    for (int i=1; i<nx; i++) {
        for (int j=1; j<ny; j++) {
            for (int k=1; k<nz; k++) {
                // neighbor cubeTags of a hexahedron cube
                // assume PBC default
                cubeTag    = ijk2cubeTags(xidxs(i  ), yidxs(j  ), zidxs(k  ));
                cubeTag_xM = ijk2cubeTags(xidxs(i-1), yidxs(j  ), zidxs(k  ));
                cubeTag_xP = ijk2cubeTags(xidxs(i+1), yidxs(j  ), zidxs(k  ));
                cubeTag_yM = ijk2cubeTags(xidxs(i  ), yidxs(j-1), zidxs(k  ));
                cubeTag_yP = ijk2cubeTags(xidxs(i  ), yidxs(j+1), zidxs(k  ));
                cubeTag_zM = ijk2cubeTags(xidxs(i  ), yidxs(j  ), zidxs(k-1));
                cubeTag_zP = ijk2cubeTags(xidxs(i  ), yidxs(j  ), zidxs(k+1));
                
                // EtoE: neighbor elemTag of 4 faces in a tetrahedron
                // element 0
                // face 0 : (7,0,3) <-> (0,7,3) elem1, face1, rot0
                EtoE(elemTag  ,0) = elemTag+1;
                EtoF(elemTag  ,0) = 1;
                EtoR(elemTag  ,0) = 0;
                // face 1 : (0,7,1) <-> (7,0,1) elem5, face0, rot0
                EtoE(elemTag  ,1) = elemTag+5;
                EtoF(elemTag  ,1) = 0;
                EtoR(elemTag  ,1) = 0;
                // face 2 : (7,3,1) <-> nodeTag_zP (6,2,0) -> (6,0,2) elem2, face3, rot1
                EtoE(elemTag  ,2) = (cubeTag_zP-cubeTag)*6 + elemTag+2;
                EtoF(elemTag  ,2) = 3;
                EtoR(elemTag  ,2) = 1;
                // face 3 : (3,0,1) <-> nodeTag_xM (7,4,5) -> (7,5,4) elem4, face2, rot1
                EtoE(elemTag  ,3) = (cubeTag_xM-cubeTag)*6 + elemTag+4;
                EtoF(elemTag  ,3) = 2;
                EtoR(elemTag  ,3) = 1;

                // element 1
                // face 0 : (7,0,2) <-> (0,7,2) elem2, face1, rot0
                EtoE(elemTag+1,0) = elemTag+2;
                EtoF(elemTag+1,0) = 1;
                EtoR(elemTag+1,0) = 0;
                // face 1 : (0,7,3) <-> (7,0,3) elem0, face0, rot0
                EtoE(elemTag+1,1) = elemTag+0;
                EtoF(elemTag+1,1) = 0;
                EtoR(elemTag+1,1) = 0;
                // face 2 : (7,2,3) <-> nodeTag_yP (5,0,1) -> (1,0,5) elem5, face3, rot2
                EtoE(elemTag+1,2) = (cubeTag_yP-cubeTag)*6 + elemTag+5;
                EtoF(elemTag+1,2) = 3;
                EtoR(elemTag+1,2) = 2;
                // face 3 : (2,0,3) <-> nodeTag_xM (6,4,7) -> (7,4,6) elem3, face2, rot2
                EtoE(elemTag+1,3) = (cubeTag_xM-cubeTag)*6 + elemTag+3;
                EtoF(elemTag+1,3) = 2;
                EtoR(elemTag+1,3) = 2;

                // element 2
                // face 0 : (7,0,6) <-> (0,7,6) elem3, face1, rot0
                EtoE(elemTag+2,0) = elemTag+3;
                EtoF(elemTag+2,0) = 1;
                EtoR(elemTag+2,0) = 0;
                // face 1 : (0,7,2) <-> (7,0,2) elem1, face0, rot0
                EtoE(elemTag+2,1) = elemTag+1;
                EtoF(elemTag+2,1) = 0;
                EtoR(elemTag+2,1) = 0;
                // face 2 : (7,6,2) <-> nodeTag_yP (5,4,0) -> (5,0,4) elem4, face3, rot1
                EtoE(elemTag+2,2) = (cubeTag_yP-cubeTag)*6 + elemTag+4;
                EtoF(elemTag+2,2) = 3;
                EtoR(elemTag+2,2) = 1;
                // face 3 : (6,0,2) <-> nodeTag_zM (7,1,3) -> (7,3,1) elem0, face2, rot1
                EtoE(elemTag+2,3) = (cubeTag_zM-cubeTag)*6 + elemTag+0;
                EtoF(elemTag+2,3) = 2;
                EtoR(elemTag+2,3) = 1;

                // element 3
                // face 0 : (7,0,4) <-> (0,7,4) elem4, face1, rot0
                EtoE(elemTag+3,0) = elemTag+4;
                EtoF(elemTag+3,0) = 1;
                EtoR(elemTag+3,0) = 0;
                // face 1 : (0,7,6) <-> (7,0,6) elem2, face0, rot0
                EtoE(elemTag+3,1) = elemTag+2;
                EtoF(elemTag+3,1) = 0;
                EtoR(elemTag+3,1) = 0;
                // face 2 : (7,4,6) <-> nodeTag_xP (3,0,2) -> (2,0,3) elem1, face3, rot2
                EtoE(elemTag+3,2) = (cubeTag_xP-cubeTag)*6 + elemTag+1;
                EtoF(elemTag+3,2) = 3;
                EtoR(elemTag+3,2) = 2;
                // face 3 : (4,0,6) <-> nodeTag_zM (5,1,7) -> (7,1,5) elem5, face2, rot2
                EtoE(elemTag+3,3) = (cubeTag_zM-cubeTag)*6 + elemTag+5;
                EtoF(elemTag+3,3) = 2;
                EtoR(elemTag+3,3) = 2;

                // element 4
                // face 0 : (7,0,5) <-> (0,7,5) elem5, face1, rot0
                EtoE(elemTag+4,0) = elemTag+5;
                EtoF(elemTag+4,0) = 1;
                EtoR(elemTag+4,0) = 0;
                // face 1 : (0,7,4) <-> (7,0,4) elem3, face0, rot0
                EtoE(elemTag+4,1) = elemTag+3;
                EtoF(elemTag+4,1) = 0;
                EtoR(elemTag+4,1) = 0;
                // face 2 : (7,5,4) <-> nodeTag_xP (3,1,0) -> (3,0,1) elem0, face3, rot1
                EtoE(elemTag+4,2) = (cubeTag_xP-cubeTag)*6 + elemTag+0;
                EtoF(elemTag+4,2) = 3;
                EtoR(elemTag+4,2) = 1;
                // face 3 : (5,0,4) <-> nodeTag_yM (7,2,6) -> (7,6,2) elem2, face2, rot1
                EtoE(elemTag+4,3) = (cubeTag_yM-cubeTag)*6 + elemTag+2;
                EtoF(elemTag+4,3) = 2;
                EtoR(elemTag+4,3) = 1;

                // element 5
                // face 0 : (7,0,1) <-> (0,7,1) elem0, face1, rot0
                EtoE(elemTag+5,0) = elemTag+0;
                EtoF(elemTag+5,0) = 1;
                EtoR(elemTag+5,0) = 0;
                // face 1 : (0,7,5) <-> (7,0,5) elem4, face0, rot0
                EtoE(elemTag+5,1) = elemTag+4;
                EtoF(elemTag+5,1) = 0;
                EtoR(elemTag+5,1) = 0;
                // face 2 : (7,1,5) <-> nodeTag_zP (6,0,4) -> (4,0,6) elem3, face3, rot2
                EtoE(elemTag+5,2) = (cubeTag_zP-cubeTag)*6 + elemTag+3;
                EtoF(elemTag+5,2) = 3;
                EtoR(elemTag+5,2) = 2;
                // face 3 : (1,0,5) <-> nodeTag_yM (3,2,7) -> (7,2,3) elem1, face2, rot2
                EtoE(elemTag+5,3) = (cubeTag_yM-cubeTag)*6 + elemTag+1;
                EtoF(elemTag+5,3) = 2;
                EtoR(elemTag+5,3) = 2;

                //
                // non-PBC faces
                //
                
                if (phyTags(0) >= 0) {
                    // -x (xM)
                    if (i == 1) {
                        EtoE(elemTag  ,3) = -99;
                        EtoE(elemTag+1,3) = -99;
                        EtoB(elemTag  ,3) = phyTags(0);
                        EtoB(elemTag+1,3) = phyTags(0);
                    }
                    // +x (xP)
                    else if (i == nx-1) {
                        EtoE(elemTag+3,2) = -99;
                        EtoE(elemTag+4,2) = -99;
                        EtoB(elemTag+3,2) = phyTags(0);
                        EtoB(elemTag+4,2) = phyTags(0);
                    }
                }
                if (phyTags(1) >= 0) {
                    // -y (yM) 
                    if (j == 1) {
                        EtoE(elemTag+4,3) = -99;
                        EtoE(elemTag+5,3) = -99;
                        EtoB(elemTag+4,3) = phyTags(1);
                        EtoB(elemTag+5,3) = phyTags(1);
                    }
                    // +y (yP) 
                    else if (j == ny-1) {
                        EtoE(elemTag+1,2) = -99;
                        EtoE(elemTag+2,2) = -99;
                        EtoB(elemTag+1,2) = phyTags(1);
                        EtoB(elemTag+2,2) = phyTags(1);
                    }
                }
                if (phyTags(2) >= 0) {
                    // -z (zM) 
                    if (k == 1) {
                        EtoE(elemTag+2,3) = -99;
                        EtoE(elemTag+3,3) = -99;
                        EtoB(elemTag+2,3) = phyTags(2);
                        EtoB(elemTag+3,3) = phyTags(2);
                    }
                    // +z (zP)
                    else if (k == nz-1) {
                        EtoE(elemTag  ,2) = -99;
                        EtoE(elemTag+5,2) = -99;
                        EtoB(elemTag  ,2) = phyTags(2);
                        EtoB(elemTag+5,2) = phyTags(2);
                    }
                }

                elemTag += 6;
            }
        }
    }
}
                

PYBIND11_MODULE(uniform_3d_single_cpp, mod) {
    mod.def("update_EtoV", &update_EtoV, "");
    mod.def("update_EtoEFRB", &update_EtoEFRB, "");
}
