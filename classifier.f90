SUBROUTINE classifier(weights, image, classify)
    IMPLICIT NONE
    DOUBLE PRECISION, INTENT(IN) :: image(28 * 28), weights(28 * 28, 512, 8)
    DOUBLE PRECISION, INTENT(OUT) :: classify(10)
    DOUBLE PRECISION :: A(28 * 28, 512), Y(512)
    EXTERNAL :: DGEMV

    A = weights(:, :, 1)
    Y = weights(1:512, 1, 3)
    call DGEMV('T', 28 * 28, 512, 1.0d0, A, 28 * 28, image, 1, 1.0d0, Y, 1)

    A(1:512, 1:10) = weights(1:512, 1:10, 2)
    Y = MAX(0, Y)
    classify = weights(1:10, 1, 4)
    call DGEMV('T', 512, 10, 1.0d0, A, 28 * 28, Y, 1, 1.0d0, classify, 1)
END
